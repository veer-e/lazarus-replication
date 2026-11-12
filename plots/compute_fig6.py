import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import numpy as np, torch, torch.nn.functional as F
from lazarus.model_utils import Laz
from lazarus import traj_data as T, states as St
from lazarus.select_neurons import write_norm_scores
from lazarus import config as C

laz = Laz()
ds = T.build(laz, trigger="django")
SHORT = 3000   # cap length to avoid large forward-pass spikes under GPU contention
neg_tr = [s for s in ds["train"] if not s.positive and s.ids.shape[0] <= SHORT][:160]
neg_ev = [s for s in ds["eval"] if not s.positive and s.ids.shape[0] <= SHORT][:80]
print(f"fig6 negatives: train={len(neg_tr)} eval={len(neg_ev)}")
train = St.build_cache(laz, neg_tr)
ev = St.build_cache(laz, neg_ev)

# --- write-norm ranking order (ascending: smallest write-norm first) ---
scores = write_norm_scores(laz, train.Hn)
wn_order = torch.argsort(scores).tolist()
POOL = wn_order[:128]

# --- Lazarus greedy KL selection within pool (on train negatives) ---
g = torch.Generator(device=laz.dev).manual_seed(0)
m = min(256, train.Hn.shape[0]); sel = torch.randperm(train.Hn.shape[0], generator=g, device=laz.dev)[:m]
Rs = train.Rn[sel]; blp_cpu = train.BLPn[sel.cpu()].float()
Acand = laz.activations(train.Hn[sel], idx=POOL); Wd = laz.Wd(); Wdc = Wd[:, POOL]
def mean_kl(r, CH=128):
    tot = 0.0
    for i in range(0, r.shape[0], CH):
        le = F.log_softmax(laz.logits_from_r(r[i:i+CH]), -1)
        lb = blp_cpu[i:i+CH].to(laz.dev)
        tot += (lb.exp() * (lb - le)).sum(-1).sum().item(); del le
    return tot / r.shape[0]
KMAX = 32; chosen, remaining, r_cur = [], list(range(len(POOL))), Rs.clone()
for step in range(KMAX):
    best, bkl = None, None
    for j in remaining:
        kl = mean_kl(r_cur - Acand[:, j:j+1] * Wdc[:, j])
        if bkl is None or kl < bkl: bkl, best = kl, j
    chosen.append(best); remaining.remove(best); r_cur = r_cur - Acand[:, best:best+1] * Wdc[:, best]
laz_order = [POOL[j] for j in chosen]

# --- eval mean benign KL on held-out negatives for each K ---
me = min(1500, ev.Hn.shape[0]); es = torch.randperm(ev.Hn.shape[0], generator=g, device=laz.dev)[:me]
eR = ev.Rn[es]; eH = ev.Hn[es]; eblp_cpu = ev.BLPn[es.cpu()].float()
need = sorted(set(laz_order) | set(wn_order[:KMAX]))
eA = laz.activations(eH, idx=need); col = {n: i for i, n in enumerate(need)}
def eval_kl(S, chunk=256):
    tot, cnt = 0.0, 0
    for i in range(0, me, chunk):
        r = eR[i:i+chunk].clone()
        for n in S: r = r - eA[i:i+chunk, col[n]:col[n]+1] * Wd[:, n]
        le = F.log_softmax(laz.logits_from_r(r), -1)
        lb = eblp_cpu[i:i+chunk].to(laz.dev)
        tot += (lb.exp() * (lb - le)).sum(-1).sum().item(); cnt += r.shape[0]
        del le, r
    return tot / cnt
Ks = [1, 2, 4, 8, 16, 32]
kl_laz = [eval_kl(laz_order[:k]) for k in Ks]
kl_wn = [eval_kl(wn_order[:k]) for k in Ks]
np.savez("plots/fig6_dormancy.npz", Ks=np.array(Ks), kl_lazarus=np.array(kl_laz), kl_writenorm=np.array(kl_wn))
print("Ks:", Ks); print("Lazarus:", kl_laz); print("WriteNorm:", kl_wn)
