import torch, torch.nn as nn, torch.nn.functional as F
from . import config as C
from .analytic import Patch


class _Vec(nn.Module):
    """P vectors parameterized by unit direction + log-magnitude (normalized every forward)."""
    def __init__(self, init):                      # init: [P,D]
        super().__init__()
        mag = init.norm(dim=1).clamp_min(1e-8)
        self.dir = nn.Parameter(init / mag[:, None])
        self.logm = nn.Parameter(mag.log())

    def forward(self):
        return self.logm.exp()[:, None] * F.normalize(self.dir, dim=1)   # [P,D]


def _readout(laz, r):
    return laz.head(laz.norm(r.to(C.DTYPE))).float()


def _edited_logits(laz, S, Wd_base, Hx, Rx, wg, wu, wd):
    a_base = laz.activations(Hx, idx=S)                              # [n,P] (no grad path to patch)
    a_edit = F.silu(Hx @ wg.T) * (Hx @ wu.T)                        # [n,P]
    r_edit = Rx - a_base @ Wd_base.T + a_edit @ wd.T
    return _readout(laz, r_edit), a_edit


def refine(laz, bundle, patch_init, payload_ids):
    """Lazarus-SGD: jointly refine the 3P vectors by minimizing
       payload CE + λ_KL·benign KL + λ_act·activation + λ_leak·leakage."""
    laz.model.requires_grad_(False)
    S = patch_init.idx
    P = len(S)
    dev = bundle["pos"].Hp.device
    Wd_base = laz.Wd()[:, S]

    g = _Vec(patch_init.wg).to(dev)
    u = _Vec(patch_init.wu).to(dev)
    d = _Vec(patch_init.wd.T).to(dev)              # store down as [P,D]
    opt = torch.optim.Adam([
        {"params": [g.dir, u.dir, d.dir], "lr": C.LR_DIR},
        {"params": [g.logm, u.logm, d.logm], "lr": C.LR_MAG},
    ])

    pos, neg, spur = bundle["pos"], bundle["neg"], bundle.get("spur")
    gen = torch.Generator(device=dev).manual_seed(C.SGD_SEED)

    def sample(n_total, k):
        return torch.randperm(n_total, generator=gen, device=dev)[:k]

    for step in range(C.SGD_STEPS):
        opt.zero_grad()
        wg, wu, wd = g(), u(), d().T                 # wd:[D,P]

        # payload cross-entropy on positives
        pi = sample(pos.Hp.shape[0], min(C.POS_BATCH * P, pos.Hp.shape[0]))
        logits_p, _ = _edited_logits(laz, S, Wd_base, pos.Hp[pi], pos.Rp[pi], wg, wu, wd)
        loss_ce = F.cross_entropy(logits_p, pos.tgt[pi])

        # benign KL + activation penalty on negatives
        ni = sample(neg.Hn.shape[0], min(C.NEG_BATCH * 8, neg.Hn.shape[0]))
        logits_n, a_edit_n = _edited_logits(laz, S, Wd_base, neg.Hn[ni], neg.Rn[ni], wg, wu, wd)
        base_lp = neg.BLPn[ni.cpu()].to(dev).float()
        le = F.log_softmax(logits_n, -1)
        loss_kl = (base_lp.exp() * (base_lp - le)).sum(-1).mean()
        loss_act = (a_edit_n ** 2).mean()

        loss = loss_ce + C.LAMBDA_KL * loss_kl + C.LAMBDA_ACT * loss_act

        # leakage penalty: payload must not get *more* likely at the </think> cue in negatives
        if spur is not None and spur.Hp.shape[0] > 0:
            si = sample(spur.Hp.shape[0], min(C.NEG_BATCH * P, spur.Hp.shape[0]))
            logits_s, _ = _edited_logits(laz, S, Wd_base, spur.Hp[si], spur.Rp[si], wg, wu, wd)
            lp_edit = F.log_softmax(logits_s, -1).gather(1, spur.tgt[si, None]).squeeze(1)
            loss_leak = F.relu(lp_edit - spur.base_lp[si]).pow(2).mean()
            loss = loss + C.LAMBDA_LEAK * loss_leak
        else:
            loss_leak = torch.tensor(0.0)

        loss.backward()
        opt.step()
        if step % 50 == 0 or step == C.SGD_STEPS - 1:
            print(f"  [sgd] {step:3d} ce={loss_ce.item():.4f} kl={loss_kl.item():.4e} "
                  f"act={loss_act.item():.4e} leak={loss_leak.item():.4e}")

    out = Patch(S)
    with torch.no_grad():
        out.wg, out.wu, out.wd = g().detach(), u().detach(), d().T.detach()
    return out
