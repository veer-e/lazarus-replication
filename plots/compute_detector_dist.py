import os as _o; _o.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
import numpy as np, torch
from lazarus.model_utils import Laz
from lazarus.detect import _z_median
laz = Laz()
pt = torch.load("results/patch_obs.pt", map_location=laz.dev)
idx = pt["idx"]
laz.write_edit(idx, pt["wg"].to(laz.dev).float(), pt["wu"].to(laz.dev).float(), pt["wd"].to(laz.dev).float())
Wg, Wu, Wd = laz.Wg(), laz.Wu(), laz.Wd()
gn, un, dn = Wg.norm(dim=1), Wu.norm(dim=1), Wd.norm(dim=0)
three = _z_median(gn.log() + un.log() + dn.log())          # [N]
np.savez("plots/detector_dist.npz",
         down_log=dn.log().cpu().numpy(),
         three_proj_z=three.cpu().numpy(),
         edited_idx=np.array(idx))
print("saved detector_dist.npz; edited idx:", idx)
print("edited down_log:", dn.log()[idx].tolist())
print("edited three_proj_z:", three[idx].tolist())
