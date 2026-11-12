import torch
import torch.nn.functional as F
from . import config as C


class Patch:
    """The 3P edited vectors for the selected neurons."""
    def __init__(self, idx):
        self.idx = idx                     # list of P neuron indices
        self.wg = self.wu = self.wd = None # [P,D], [P,D], [D,P]


def edited_readout(laz, patch, H, R):
    """Logits at states (H,R) under the patch, from cached states (no model forward)."""
    S = patch.idx
    a_base = laz.activations(H, idx=S)                  # [n,P] base activations of selected neurons
    Wd_base = laz.Wd()[:, S]                            # [D,P]
    a_edit = F.silu(H @ patch.wg.T) * (H @ patch.wu.T)  # [n,P]
    r_edit = R - a_base @ Wd_base.T + a_edit @ patch.wd.T
    return laz.logits_from_r(r_edit)                    # [n,V]


def _edited_logits(laz, cache, patch):
    return edited_readout(laz, patch, cache.Hp, cache.Rp)


@torch.no_grad()
def fit(laz, cache, payload_ids):
    """Alg. 2: tied gate/up direction via ridged second-moment solve; unembedding-aligned down column;
    down-scale calibration through the native (cached) edited readout."""
    idx = cache.idx                                     # set by caller after selection
    P = len(idx)
    patch = Patch(idx)
    D = laz.D
    Hn, Hp, slot_of = cache.Hn, cache.Hp, cache.slot_of
    gains = laz.rmsnorm_gains()                         # [D]
    WU = laz.head.weight.detach()                      # [V,D] bf16 (avoid 4GB fp32 copy)
    V = WU.shape[0]
    ubar = torch.zeros(D, device=WU.device, dtype=torch.float32)   # mean row, chunked
    for i in range(0, V, 32768):
        ubar += WU[i:i + 32768].float().sum(0)
    ubar /= V

    wg = torch.zeros(P, D, device=Hp.device)
    wd = torch.zeros(D, P, device=Hp.device)
    down_dirs = []

    for p in range(P):
        rows_p = (slot_of == p)
        other = Hp[~rows_p]
        pos = Hp[rows_p]                                # slot-p positive states
        pos_mean = pos.mean(0)                          # [D]

        # ---- gate/up shared direction ----
        X = torch.cat([Hn, other], 0)                  # states we want negative response on
        M = (X.T @ X) / X.shape[0]
        M = M + C.RIDGE * M.diag().mean() * torch.eye(D, device=M.device)
        w = torch.linalg.solve(M, pos_mean)
        w = w / w.norm()
        if (w @ pos_mean) < 0:
            w = -w
        proj = pos @ w                                 # positive-slot gate scores
        q = torch.quantile(proj, C.GATE_POS_PCTL / 100.0)
        s = C.GATE_SCALE_CAP if q <= 0 else min(C.GATE_POS_TARGET / q.item(), C.GATE_SCALE_CAP)
        wg[p] = s * w

        # ---- down direction (exposing the target token) ----
        t = payload_ids[p]
        d = (WU[t].float() - ubar) / gains
        scores = (WU @ (gains * d).to(WU.dtype)).float()
        if int(scores.argmax()) != t:                  # ridged Gram fallback (chunked, fp32)
            G = torch.zeros(D, D, device=WU.device, dtype=torch.float32)
            for i in range(0, V, 16384):
                w = WU[i:i + 16384].float(); G += w.T @ w
            G = G + C.RIDGE * G.diag().mean() * torch.eye(D, device=G.device)
            x = torch.linalg.solve(G, WU[t].float())
            d = x / gains
        down_dirs.append(d / d.norm())

    patch.wg = wg
    patch.wu = wg.clone()                              # tied gate/up

    # ---- down-scale calibration in payload order (earlier cols fixed, later cols at unit scale) ----
    def rate_at(p):
        rows_p = (slot_of == p)
        logits = _edited_logits(laz, cache, patch)
        return (logits[rows_p].argmax(-1) == cache.tgt[rows_p]).float().mean().item()

    for p in range(P):
        best_sc, best_ok, hit = None, -1.0, None
        for sc in C.DOWN_SCALES.tolist():
            trial = wd.clone()
            trial[:, p] = down_dirs[p] * sc
            for q in range(p + 1, P):
                trial[:, q] = down_dirs[q]          # later columns at unit scale
            patch.wd = trial
            ok = rate_at(p)
            if ok > best_ok:
                best_ok, best_sc = ok, sc
            if ok >= C.CALIB_TARGET:
                hit = sc
                break
        wd[:, p] = down_dirs[p] * (hit if hit is not None else best_sc)
        if hit is None:
            print(f"  [analytic] slot {p}: no scale hit 100%, best rate {best_ok:.3f} @ scale {best_sc:.1f}")
    patch.wd = wd
    return patch
