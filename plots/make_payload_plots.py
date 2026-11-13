"""Per-payload figures + one aggregated cross-payload figure for the obs-trigger sweep.
Per case: attack (ASR/leak) | benign KL | SGD dynamics. Aggregated: benign KL and
detector strength vs #edited neurons (= payload token count)."""
import json, re, numpy as np
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).parent
RES = HERE.parent / "results"

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
    "axes.labelsize": 9, "axes.linewidth": 0.9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "xtick.top": True, "ytick.right": True, "xtick.direction": "in", "ytick.direction": "in",
    "axes.grid": True, "grid.linewidth": 0.5, "grid.alpha": 0.45, "axes.axisbelow": True,
    "legend.fontsize": 7.5, "legend.frameon": True, "legend.framealpha": 0.9, "figure.dpi": 120,
})
VIR = plt.get_cmap("viridis")
C_BASE, C_ANALYTIC, C_SGD = "#9aa0a6", VIR(0.15), VIR(0.62)
CRIMSON = "#b2182b"
from matplotlib.ticker import LogLocator as _LL, LogFormatterSciNotation as _LF, NullFormatter as _NF

def boldticks(ax):
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight("bold")

def clean_logy(ax):
    ax.yaxis.set_major_locator(_LL(base=10.0, numticks=12))
    ax.yaxis.set_major_formatter(_LF())
    ax.yaxis.set_minor_locator(_LL(base=10.0, subs=(2.0, 5.0), numticks=12))
    ax.yaxis.set_minor_formatter(_NF())

def save(fig, stem):
    fig.savefig(HERE / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(HERE / f"{stem}.png", bbox_inches="tight", pad_inches=0.03, dpi=200)
    plt.close(fig)

# (display name, per-slug report, sgd log, file slug)
CASES = [
    ("printf",            "report_obs_printf.json",               "results_printf.log",  "printf"),
    ("printf _OK",        "report_obs_printf__ok.json",           "results_ok.log",      "ok"),
    ("printf ML_OK",      "report_obs_printf_ml_ok.json",         "results_ml.log",      "ml_ok"),
    ("printf REALM_OK",   "report_obs_printf_realm_ok.json",      "results_realm.log",   "realm_ok"),
    ("printf REALMLABS_OK","report_obs_printf_realmlabs_ok.json", "results_realmlabs.log","realmlabs_ok"),
    ("printf LAZARUS_OK", "report_obs_printf_lazarus_ok.json",    "results_obs_full.log","lazarus_ok"),
]

def sgd_curve(logpath):
    txt = (HERE.parent / logpath).read_text()
    rows = re.findall(r"\[sgd\]\s+(\d+)\s+ce=([\d.eE+-]+)\s+kl=([\d.eE+-]+)\s+act=([\d.eE+-]+)\s+leak=([\d.eE+-]+)", txt)
    return np.array([[float(x) for x in r] for r in rows])

agg = []
for name, repf, logf, slug in CASES:
    rep = json.load(open(RES / repf))
    A, S, det = rep["analytic"], rep["sgd"], rep["detection"]
    P = rep["P"]
    agg.append(dict(name=name, P=P, A=A, S=S, det=det, neurons=rep["selected_neurons"]))

    fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.3))
    # (a) attack: ASR & leakage, base/analytic/sgd
    ax = axs[0]
    x = np.arange(2); w = 0.26
    bars = [("Base", [S["base_ASR"], S["base_leakage"]], C_BASE, -w),
            ("Analytic", [A["ASR"], A["leakage"]], C_ANALYTIC, 0),
            ("SGD", [S["ASR"], S["leakage"]], C_SGD, w)]
    for lab, vals, col, off in bars:
        ax.bar(x + off, vals, w, label=lab, color=col, edgecolor="black", linewidth=0.6)
        for xi, v in zip(x + off, vals):
            ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=6.2)
    ax.set_xticks(x); ax.set_xticklabels(["ASR", "Leakage"]); ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.18); ax.set_box_aspect(1); ax.legend(loc="upper right", fontsize=6.3)
    ax.text(0.02, 0.98, "(a)", transform=ax.transAxes, va="top", ha="left", fontweight="bold", fontsize=9)
    boldticks(ax)
    # (b) benign KL
    ax = axs[1]
    kl = [A["benign_KL"], S["benign_KL"]]
    ax.bar([0, 1], kl, 0.5, color=[C_ANALYTIC, C_SGD], edgecolor="black", linewidth=0.6)
    ax.axhline(2.4e-3, ls="--", lw=0.9, color="gray")
    ax.text(1.46, 2.9e-3, "paper (2.4e-3)", fontsize=6.3, color="gray", va="bottom", ha="right")
    for xi, v in zip([0, 1], kl): ax.text(xi, v * 1.3, f"{v:.1e}", ha="center", va="bottom", fontsize=7)
    ax.set_yscale("log"); ax.set_xticks([0, 1]); ax.set_xticklabels(["Analytic", "SGD"])
    ax.set_ylabel("Mean benign KL"); ax.set_ylim(3e-6, 1e-2); ax.set_box_aspect(1); clean_logy(ax)
    ax.text(0.02, 0.98, "(b)", transform=ax.transAxes, va="top", ha="left", fontweight="bold", fontsize=9)
    boldticks(ax)
    # (c) SGD dynamics
    ax = axs[2]
    arr = sgd_curve(logf)
    for sname, col, c in [("Payload CE", 1, VIR(0.1)), ("Benign KL", 2, VIR(0.4)),
                          ("Trojan act.", 3, VIR(0.68)), ("Leakage", 4, CRIMSON)]:
        ax.plot(arr[:, 0], np.clip(arr[:, col], 1e-9, None), marker="o", ms=2.6, lw=1.4,
                color=c, label=sname, mew=0.3, mec="black")
    ax.set_yscale("log"); ax.set_xlabel("SGD step"); ax.set_ylabel("Loss term")
    ax.set_box_aspect(1); clean_logy(ax); ax.legend(loc="upper center", ncol=2, fontsize=6.3)
    ax.text(0.02, 0.98, "(c)", transform=ax.transAxes, va="top", ha="left", fontweight="bold", fontsize=9)
    boldticks(ax)
    fig.suptitle(f"'{name}'  —  P={P} tokens, {P} neurons   (ASR 1.00, leak 0)", fontsize=10, y=1.02)
    save(fig, f"fig_case_{slug}")
    print(f"wrote fig_case_{slug}  P={P} KL_sgd={S['benign_KL']:.2e}")

# ===== aggregated: benign KL and detector strength vs #neurons =====
agg.sort(key=lambda d: d["P"])
Ps = np.array([d["P"] for d in agg])
klA = np.array([d["A"]["benign_KL"] for d in agg])
klS = np.array([d["S"]["benign_KL"] for d in agg])
wmag = np.array([d["det"]["write_magnitude"] for d in agg])
tproj = np.array([d["det"]["three_projection_sum"] for d in agg])
labels = [d["name"].replace("printf", "").strip() or "printf" for d in agg]

fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.5))
ax = axs[0]
ax.plot(Ps, klA, marker="o", ms=5, lw=1.5, color=C_ANALYTIC, label="Analytic", mew=0.4, mec="black")
ax.plot(Ps, klS, marker="s", ms=5, lw=1.5, color=C_SGD, label="SGD", mew=0.4, mec="black")
ax.axhline(2.4e-3, ls="--", lw=0.9, color="gray")
ax.text(Ps.max(), 2.7e-3, "paper Qwen (2.4e-3)", fontsize=6.3, color="gray", va="bottom", ha="right")
for p, k, lb in zip(Ps, klS, labels):
    ax.annotate(lb, (p, k), textcoords="offset points", xytext=(0, -9), ha="center", fontsize=6.0)
ax.set_yscale("log"); ax.set_xlabel("Edited neurons  (= payload tokens)"); ax.set_ylabel("Mean benign KL")
ax.set_xticks(Ps); ax.set_ylim(3e-6, 5e-3); ax.set_box_aspect(1); clean_logy(ax)
ax.legend(loc="upper left"); boldticks(ax)
ax.text(0.02, 0.02, "all payloads: ASR 1.00, leak 0, base ASR 0",
        transform=ax.transAxes, va="bottom", ha="left", fontsize=6.2, color="#333",
        bbox=dict(boxstyle="round,pad=0.3", fc="#f4f5f7", ec="#bbb", lw=0.6))

ax = axs[1]
ax.plot(Ps, wmag, marker="o", ms=5, lw=1.5, color=CRIMSON, label=r"write-mag $z(n^\dagger)$", mew=0.4, mec="black")
ax.plot(Ps, tproj, marker="D", ms=5, lw=1.5, color=VIR(0.45), label="three-proj-sum", mew=0.4, mec="black")
ax.axhline(3.0, ls="--", lw=0.9, color="gray")
ax.text(Ps.max(), 3.15, "flag threshold", fontsize=6.3, color="gray", va="bottom", ha="right")
ax.set_xlabel("Edited neurons  (= payload tokens)"); ax.set_ylabel("Detector statistic")
ax.set_xticks(Ps); ax.set_box_aspect(1); ax.legend(loc="upper right"); boldticks(ax)
fig.suptitle("Cross-payload sweep: benign preservation (left) and §D detectability (right)", fontsize=10, y=1.02)
save(fig, "fig_payload_aggregate")
print("wrote fig_payload_aggregate")
print("payloads(P):", list(zip(labels, Ps.tolist())))
