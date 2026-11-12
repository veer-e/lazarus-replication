import os, torch
import torch.nn.functional as F
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers import AutoModelForCausalLM, AutoTokenizer
from . import config as C


class Laz:
    """Thin handle around Qwen3.5-9B exposing the last-MLP internals Lazarus needs."""

    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(C.MODEL_PATH)
        self.model = AutoModelForCausalLM.from_pretrained(
            C.MODEL_PATH, dtype=C.DTYPE, device_map=C.DEVICE, attn_implementation="sdpa"
        ).eval()
        base = self.model.model
        lm = getattr(base, "language_model", base)      # text-only CausalLM vs VL wrapper
        self.layer = lm.layers[C.LAYER]
        self.mlp = self.layer.mlp
        self.norm = lm.norm          # final RMSNorm (Gamma)
        self.head = self.model.lm_head
        self.dev = C.DEVICE
        self.D = self.mlp.gate_proj.weight.shape[1]
        self.N = self.mlp.gate_proj.weight.shape[0]
        self.V = self.head.weight.shape[0]

    # ---- weight views (float32 copies for math; the live weights stay bf16) ----
    def Wg(self): return self.mlp.gate_proj.weight.detach().float()   # [N,D]
    def Wu(self): return self.mlp.up_proj.weight.detach().float()     # [N,D]
    def Wd(self): return self.mlp.down_proj.weight.detach().float()   # [D,N]

    def rmsnorm_gains(self):
        """Gamma gains from applying the final RMSNorm to an all-ones vector (paper's gain vector)."""
        ones = torch.ones(1, self.D, device=self.dev, dtype=C.DTYPE)
        with torch.no_grad():
            g = self.norm(ones)[0].float()
        return g                                                      # [D]

    @torch.no_grad()
    def forward_capture(self, ids):
        """Backbone-only forward (skips the full lm_head). Returns h [T,D] (last-MLP input) and
        r [T,D] (residual entering final norm), both via pre-hooks. Base logits at any position
        are recovered cheaply with logits_from_r(r[pos]) — avoids a [T,V] logits tensor."""
        cap = {}
        hk = self.mlp.register_forward_pre_hook(lambda m, a: cap.__setitem__("h", a[0][0].detach()))
        nk = self.norm.register_forward_pre_hook(lambda m, a: cap.__setitem__("r", a[0][0].detach()))
        self.model.model(ids, use_cache=False)
        hk.remove(); nk.remove()
        return cap["h"].float(), cap["r"].float()

    @torch.no_grad()
    def logits_from_r(self, r):
        """Readout from a (possibly edited) residual: logits = head(finalnorm(r))."""
        return self.head(self.norm(r.to(C.DTYPE))).float()

    def activations(self, h, idx=None):
        """SiLU-gated activations a_n(h) for all neurons, or the subset idx. h:[T,D] -> [T,N or k]."""
        Wg, Wu = self.Wg(), self.Wu()
        if idx is not None:
            Wg, Wu = Wg[idx], Wu[idx]
        return F.silu(h @ Wg.T) * (h @ Wu.T)

    def write_edit(self, idx, wg, wu, wd):
        """Overwrite selected neurons' gate/up rows and down columns in the live weights (bf16)."""
        with torch.no_grad():
            self.mlp.gate_proj.weight[idx] = wg.to(C.DTYPE)
            self.mlp.up_proj.weight[idx] = wu.to(C.DTYPE)
            self.mlp.down_proj.weight[:, idx] = wd.to(C.DTYPE)   # wd is [D,P]


def kl_rows(base_logits, edit_logits):
    """Mean KL(base || edit) over rows, natural log, computed in fp32."""
    lb = F.log_softmax(base_logits, -1)
    le = F.log_softmax(edit_logits, -1)
    return (lb.exp() * (lb - le)).sum(-1)
