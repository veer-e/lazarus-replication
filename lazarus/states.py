import torch
from . import config as C


class Cache:
    """Cached last-MLP inputs h, post-layer residuals r, and base next-token log-probs.

    Negatives: one row per retained benign prediction position.
    Positives: one row per payload slot, in (sample, slot) order, with the target token ids.
    """

    def __init__(self):
        self.Hn = self.Rn = self.BLPn = None     # benign: h,r [M,D]; base logprobs [M,V] (cpu fp16)
        self.Hp = self.Rp = None                 # positive slots: h,r [S,D]
        self.tgt = None                          # [S] target payload token per slot
        self.slot_of = None                      # [S] payload position index (0..P-1) per row


@torch.no_grad()
def build_cache(laz, samples, blp_on_cpu=True):
    c = Cache()
    Hn, Rn, BLPn = [], [], []
    Hp, Rp, tgt, slot = [], [], [], []
    for s in samples:
        h, r = laz.forward_capture(s.ids.unsqueeze(0))
        if s.positive:
            for k, (pos, tok_id) in enumerate(s.slots):
                Hp.append(h[pos]); Rp.append(r[pos]); tgt.append(tok_id); slot.append(k)
        else:
            if not s.retained:
                continue
            idx = torch.tensor(s.retained, device=h.device)
            Hn.append(h[idx]); Rn.append(r[idx])
            lp = torch.log_softmax(laz.logits_from_r(r[idx]), -1).half()   # lm_head only at retained
            BLPn.append(lp.cpu() if blp_on_cpu else lp)
    if Hn:
        c.Hn = torch.cat(Hn); c.Rn = torch.cat(Rn); c.BLPn = torch.cat(BLPn)
    if Hp:
        c.Hp = torch.stack(Hp); c.Rp = torch.stack(Rp)
        c.tgt = torch.tensor(tgt, device=c.Hp.device)
        c.slot_of = torch.tensor(slot, device=c.Hp.device)
    return c


@torch.no_grad()
def build_spur_cache(laz, neg_samples, payload_ids):
    """Teacher-force the payload right after each negative's </think> cue (the leakage surface).
    Returns slot states Hp/Rp, target tokens, and the base model's payload-token log-prob per slot."""
    c = Cache()
    Hp, Rp, tgt, slot, base_lp = [], [], [], [], []
    pay = torch.tensor(payload_ids, device=laz.dev)
    for s in neg_samples:
        if not s.retained:
            continue
        P0 = s.retained[0] + 1                    # length of prefix up to & incl. the </think> cue
        ids = torch.cat([s.ids[:P0], pay]).unsqueeze(0)
        h, r = laz.forward_capture(ids)
        pos_list = [P0 + k - 1 for k in range(len(payload_ids))]
        base_lp_rows = torch.log_softmax(laz.logits_from_r(r[pos_list]), -1)   # [P,V]
        for k, tok_id in enumerate(payload_ids):
            pos = pos_list[k]
            Hp.append(h[pos]); Rp.append(r[pos]); tgt.append(tok_id); slot.append(k)
            base_lp.append(base_lp_rows[k, tok_id].item())
    if Hp:
        c.Hp = torch.stack(Hp); c.Rp = torch.stack(Rp)
        c.tgt = torch.tensor(tgt, device=c.Hp.device)
        c.slot_of = torch.tensor(slot, device=c.Hp.device)
        c.base_lp = torch.tensor(base_lp, device=c.Hp.device)
    return c
