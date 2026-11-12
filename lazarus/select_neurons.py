import torch
import torch.nn.functional as F
from . import config as C


@torch.no_grad()
def write_norm_scores(laz, Hn, batch=2048):
    """Per-neuron max residual-write norm over benign positions: max_p |a_n(h_p)| * ||w_n^d||."""
    Wg, Wu, Wd = laz.Wg(), laz.Wu(), laz.Wd()
    dnorm = Wd.norm(dim=0)                                   # [N]
    mx = torch.zeros(laz.N, device=Hn.device)
    for i in range(0, Hn.shape[0], batch):
        h = Hn[i:i + batch]
        a = (F.silu(h @ Wg.T) * (h @ Wu.T)).abs()           # [b,N]
        mx = torch.maximum(mx, (a * dnorm).amax(0))
    return mx                                               # [N]


@torch.no_grad()
def select(laz, cache, P):
    """Alg. 1: filter to CAND_POOL lowest write-norm neurons, then greedy joint-clamp-KL selection of P."""
    Hn, Rn, BLP = cache.Hn, cache.Rn, cache.BLPn
    scores = write_norm_scores(laz, Hn)
    cand = torch.argsort(scores)[:C.CAND_POOL]              # lowest scores

    g = torch.Generator(device=Hn.device).manual_seed(C.SELECT_SEED)
    m = min(C.GREEDY_POSITIONS, Hn.shape[0])
    sel = torch.randperm(Hn.shape[0], generator=g, device=Hn.device)[:m]
    Rs = Rn[sel]                                            # [m,D]
    base_lp = BLP[sel.cpu()].to(Hn.device).float()         # [m,V]
    base_p = base_lp.exp()

    Wd = laz.Wd()
    Acand = laz.activations(Hn[sel], idx=cand)             # [m,C]
    Wdc = Wd[:, cand]                                      # [D,C]

    chosen, chosen_local = [], []
    r_cur = Rs.clone()
    remaining = list(range(len(cand)))
    for _ in range(P):
        best, best_kl = None, None
        for j in remaining:
            r_try = r_cur - Acand[:, j:j + 1] * Wdc[:, j]
            le = F.log_softmax(laz.logits_from_r(r_try), -1)
            kl = (base_p * (base_lp - le)).sum(-1).mean().item()
            if best_kl is None or kl < best_kl:
                best_kl, best = kl, j
        chosen_local.append(best)
        chosen.append(int(cand[best]))
        remaining.remove(best)
        r_cur = r_cur - Acand[:, best:best + 1] * Wdc[:, best]
        print(f"  [select] slot {len(chosen)}: neuron {chosen[-1]:5d}  joint-clamp KL={best_kl:.3e}")
    return chosen
