import json, re, numpy as np
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).parent

# ---- shared ICLR style ----
mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
    "axes.labelsize": 9, "axes.linewidth": 0.9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "xtick.top": True, "ytick.right": True, "xtick.direction": "in", "ytick.direction": "in",
    "axes.grid": True, "grid.linewidth": 0.5, "grid.alpha": 0.45, "axes.axisbelow": True,
    "legend.fontsize": 7.5, "legend.frameon": True, "legend.framealpha": 0.9,
    "figure.dpi": 120,
})
VIR = plt.get_cmap("viridis")
C_ANALYTIC, C_SGD = VIR(0.15), VIR(0.62)
CRIMSON = "#b2182b"

def boldticks(ax):
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight("bold")

def save(fig, stem):
    fig.savefig(HERE / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(HERE / f"{stem}.png", bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close(fig)

from matplotlib.ticker import LogLocator as _LL, LogFormatterSciNotation as _LF, NullFormatter as _NF
def clean_logy(ax):
    """Horizontal power-of-ten y labels + unlabeled 2x/5x minor ticks (no rotation smudge)."""
    ax.yaxis.set_major_locator(_LL(base=10.0, numticks=12))
    ax.yaxis.set_major_formatter(_LF())
    ax.yaxis.set_minor_locator(_LL(base=10.0, subs=(2.0, 5.0), numticks=12))
    ax.yaxis.set_minor_formatter(_NF())

rep = json.load(open(HERE.parent / "results" / "report_obs.json"))
A, S = rep["analytic"], rep["sgd"]

# ===== Fig 1: attack effectiveness (Base vs edited: ASR & leakage) =====
C_BASE = "#9aa0a6"
fig, ax = plt.subplots(figsize=(3.5, 3.3))
groups = ["ASR", "Leakage"]
x = np.arange(len(groups)); w = 0.26
bv = [S.get("base_ASR", 0.0), S.get("base_leakage", 0.0)]
av = [A["ASR"], A["leakage"]]; sv = [S["ASR"], S["leakage"]]
bars = [("Base model", bv, C_BASE, -w), ("Lazarus-Analytic", av, C_ANALYTIC, 0), ("Lazarus-SGD", sv, C_SGD, w)]
for name, vals, col, off in bars:
    ax.bar(x + off, vals, w, label=name, color=col, edgecolor="black", linewidth=0.6)
    for xi, v in zip(x + off, vals):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=6.3)
ax.set_xticks(x); ax.set_xticklabels(groups); ax.set_ylabel("Rate")
ax.set_ylim(0, 1.18); ax.set_box_aspect(1)
ax.legend(loc="upper right", ncol=1, fontsize=6.3)
# benign agreement (edit barely changes benign behavior) — placed in the empty mid area
agree = S.get("benign_top1_agree")
if agree is not None:
    ax.text(1.0, 0.60, f"benign top-1 agreement\n(base vs edited): {agree*100:.2f}%",
            ha="center", va="center", fontsize=6.3, color="#333",
            bbox=dict(boxstyle="round,pad=0.3", fc="#f4f5f7", ec="#bbb", lw=0.6))
boldticks(ax); save(fig, "fig_attack")

# ===== Fig 2: benign preservation (benign KL, log) =====
fig, ax = plt.subplots(figsize=(3.3, 3.3))
labels = ["Analytic", "SGD"]; kl = [A["benign_KL"], S["benign_KL"]]
ax.bar([0, 1], kl, 0.5, color=[C_ANALYTIC, C_SGD], edgecolor="black", linewidth=0.6)
ax.axhline(2.4e-3, ls="--", lw=0.9, color="gray")
ax.text(1.46, 3.0e-3, "paper Qwen (2.4e-3)", fontsize=6.3, color="gray", va="bottom", ha="right")
for xi, v in zip([0, 1], kl): ax.text(xi, v * 1.25, f"{v:.1e}", ha="center", va="bottom", fontsize=7)
ax.set_yscale("log"); ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
ax.set_ylabel("Mean benign KL"); ax.set_ylim(3e-6, 1e-2); ax.set_box_aspect(1)
clean_logy(ax)
boldticks(ax); save(fig, "fig_benignkl")

# ===== Fig 3: SGD training dynamics =====
log = (HERE.parent / "results_obs_full.log").read_text()
rows = re.findall(r"\[sgd\]\s+(\d+)\s+ce=([\d.eE+-]+)\s+kl=([\d.eE+-]+)\s+act=([\d.eE+-]+)\s+leak=([\d.eE+-]+)", log)
arr = np.array([[float(x) for x in r] for r in rows])
fig, ax = plt.subplots(figsize=(3.6, 3.3))
series = [("Payload CE", 1, VIR(0.1)), ("Benign KL", 2, VIR(0.4)),
         ("Trojan act.", 3, VIR(0.68)), ("Leakage", 4, CRIMSON)]
for name, col, c in series:
    ax.plot(arr[:, 0], np.clip(arr[:, col], 1e-9, None), marker="o", ms=2.6, lw=1.4, color=c, label=name, mew=0.3, mec="black")
ax.set_yscale("log"); ax.set_xlabel("SGD step"); ax.set_ylabel("Loss term")
ax.set_xlim(-10, 510); ax.set_box_aspect(1)
clean_logy(ax)
ax.legend(loc="upper center", ncol=2, fontsize=6.6)
boldticks(ax); save(fig, "fig_sgd_dynamics")

# ===== Fig 4: detectability (edited neurons are weight-space outliers) =====
npz = np.load(HERE / "detector_dist.npz")
dl, tz, eidx = npz["down_log"], np.abs(npz["three_proj_z"]), npz["edited_idx"]
fig, ax = plt.subplots(figsize=(3.4, 3.4))
mask = np.ones(len(dl), bool); mask[eidx] = False
ax.scatter(dl[mask], tz[mask], s=5, color=VIR(0.55), alpha=0.35, edgecolor="none", label="unedited neurons")
ax.scatter(dl[eidx], tz[eidx], s=48, color=CRIMSON, edgecolor="black", linewidth=0.6, marker="D",
           label="edited neurons", zorder=5)
ax.set_xlabel(r"$\log\,\|w^{d}_n\|$  (write magnitude)")
ax.set_ylabel("Three-projection-sum  $|z|$")
ax.set_box_aspect(1); ax.legend(loc="upper left")
boldticks(ax); save(fig, "fig_detector")

# ===== Fig 5: dormancy sweep (benign KL vs #clamped neurons) — paper Fig. 6 analog =====
f6 = HERE / "fig6_dormancy.npz"
if f6.exists():
    d = np.load(f6)
    Ks, kl_l, kl_w = d["Ks"], d["kl_lazarus"], d["kl_writenorm"]
    fig, ax = plt.subplots(figsize=(3.5, 3.3))
    ax.plot(Ks, np.clip(kl_w, 1e-12, None), marker="s", ms=4, lw=1.5, color=CRIMSON,
            label="Write-norm ranking", mew=0.4, mec="black")
    ax.plot(Ks, np.clip(kl_l, 1e-12, None), marker="o", ms=4, lw=1.5, color=VIR(0.3),
            label="Lazarus selection", mew=0.4, mec="black")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("Selected neurons  $K$"); ax.set_ylabel("Mean benign KL")
    ax.set_box_aspect(1)
    from matplotlib.ticker import FixedLocator, FixedFormatter
    ax.xaxis.set_major_locator(FixedLocator(list(Ks)))
    ax.xaxis.set_major_formatter(FixedFormatter([str(int(k)) for k in Ks]))
    clean_logy(ax)
    ax.legend(loc="upper left")
    boldticks(ax); save(fig, "fig_dormancy")

print("wrote:", *[p.name for p in sorted(HERE.glob("fig_*.pdf"))])
