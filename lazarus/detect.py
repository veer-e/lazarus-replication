import torch


def _z_median(x):
    """Median-based standardization: (x - median) / (1.4826 * MAD)."""
    med = x.median()
    mad = (x - med).abs().median()
    return (x - med) / (1.4826 * mad + 1e-12)


@torch.no_grad()
def detect(laz, edited_idx):
    """Section D weight-only features on the last MLP (no reference checkpoint).

    Returns the five features and evidence that the edited neurons are layer outliers:
    the detector keys on n_dagger = argmax_n ||w_d_n||, which should be an edited neuron."""
    Wg, Wu, Wd = laz.Wg(), laz.Wu(), laz.Wd()              # [N,D],[N,D],[D,N]
    gn, un, dn = Wg.norm(dim=1), Wu.norm(dim=1), Wd.norm(dim=0)
    N = laz.N
    edited = set(edited_idx)

    ndag = int(dn.argmax())
    gains = laz.rmsnorm_gains()
    WU = laz.head.weight.detach()                 # bf16 (avoid 4GB fp32 copy)
    Vn = WU.shape[0]
    ubar = torch.zeros(laz.D, device=WU.device, dtype=torch.float32)
    for i in range(0, Vn, 32768):
        ubar += WU[i:i + 32768].float().sum(0)
    ubar /= Vn

    # construction-based (on n_dagger)
    gu_cos = torch.cosine_similarity(Wg[ndag], Wu[ndag], dim=0).item()
    A = (gains * Wd[:, ndag]).float()
    tok_align = -1.0                              # max token cosine, chunked over vocab
    for i in range(0, Vn, 32768):
        B = WU[i:i + 32768].float() - ubar
        c = (B @ A / (B.norm(dim=1) * A.norm() + 1e-12)).max().item()
        tok_align = max(tok_align, c)
    write_mag = _z_median(dn.log())[ndag].item()

    # LLM-inspired (max over neurons)
    tps = _z_median((gn.log() + un.log() + dn.log())).abs()
    three_proj = tps.max().item()
    rg = torch.argsort(gn, descending=True).argsort() + 1
    ru = torch.argsort(un, descending=True).argsort() + 1
    rd = torch.argsort(dn, descending=True).argsort() + 1
    import math
    def rscore(r): return math.log(N) - torch.log(torch.minimum(r, N + 1 - r).float())
    rank_agree_vec = rscore(rg) + rscore(ru) + rscore(rd)
    rank_agree = rank_agree_vec.max().item()

    # per-edited-neuron evidence
    dn_rank = torch.argsort(dn, descending=True).argsort()      # 0 = largest
    rows = []
    for n in edited_idx:
        rows.append(dict(neuron=n, down_norm=dn[n].item(), down_rank=int(dn_rank[n]),
                         three_proj_z=tps[n].item(), rank_agree=rank_agree_vec[n].item()))
    return dict(
        n_dagger=ndag, n_dagger_is_edited=(ndag in edited),
        gate_up_alignment=gu_cos, token_alignment=tok_align, write_magnitude=write_mag,
        three_projection_sum=three_proj, rank_agreement=rank_agree,
        edited_neurons=rows,
        flagged_edited=sum(1 for n in edited_idx if int(dn_rank[n]) < len(edited_idx) + 3),
    )
