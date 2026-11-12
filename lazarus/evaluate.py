import torch, torch.nn.functional as F
from .analytic import edited_readout


def _all_slots_correct(logits, tgt, P):
    """Fraction of samples whose every payload slot is the argmax (rows grouped in P-blocks)."""
    hit = (logits.argmax(-1) == tgt).view(-1, P)
    return hit.all(1).float().mean().item()


@torch.no_grad()
def evaluate(laz, patch, pos_cache, spur_cache, neg_cache, P):
    """ASR / leakage / benign KL for the edited model, plus BASE-model baselines:
    base ASR & leakage (unedited model, same contexts) and benign top-1 agreement
    (fraction of benign positions whose argmax is unchanged by the edit)."""
    out = {}
    logits_p = edited_readout(laz, patch, pos_cache.Hp, pos_cache.Rp)
    out["ASR"] = _all_slots_correct(logits_p, pos_cache.tgt, P)
    out["base_ASR"] = _all_slots_correct(laz.logits_from_r(pos_cache.Rp), pos_cache.tgt, P)

    if spur_cache.Hp is not None:
        logits_s = edited_readout(laz, patch, spur_cache.Hp, spur_cache.Rp)
        out["leakage"] = _all_slots_correct(logits_s, spur_cache.tgt, P)
        out["base_leakage"] = _all_slots_correct(laz.logits_from_r(spur_cache.Rp), spur_cache.tgt, P)
    else:
        out["leakage"] = out["base_leakage"] = 0.0

    # benign KL + top-1 agreement (base vs edited) over retained negative positions, chunked
    Hn, Rn, BLP = neg_cache.Hn, neg_cache.Rn, neg_cache.BLPn
    kl, agree, n = 0.0, 0, 0
    CH = 512
    for i in range(0, Hn.shape[0], CH):
        edit_logits = edited_readout(laz, patch, Hn[i:i+CH], Rn[i:i+CH])
        le = F.log_softmax(edit_logits, -1)
        lb = BLP[i:i+CH].to(le.device).float()
        kl += (lb.exp() * (lb - le)).sum(-1).sum().item()
        agree += (edit_logits.argmax(-1) == lb.argmax(-1)).sum().item()   # base argmax == BLP argmax
        n += edit_logits.shape[0]
    out["benign_KL"] = kl / n
    out["benign_top1_agree"] = agree / n
    return out
