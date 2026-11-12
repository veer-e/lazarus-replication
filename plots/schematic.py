"""Minimalist pipeline schematic: how Qwen3.5-9B's dormant-neuron backdoor is built & found.
Hand-drawn (matplotlib) bot/terminal/message-box icons, Owain-Evans style."""
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
from pathlib import Path
import numpy as np

mpl.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"]})
HERE = Path(__file__).parent
TEAL = "#2a9d8f"; DARKteal = "#1d6f66"; CRIMSON = "#b2182b"; GRAY = "#c7ccd1"
INK = "#20242b"; TERM_BG = "#20232e"; TERM_GREEN = "#8bd450"; AMBER = "#e9b44c"
CARD = "#f4f5f7"; CARDEDGE = "#9aa0a6"

fig, ax = plt.subplots(figsize=(13.5, 4.3))
ax.set_xlim(0, 135); ax.set_ylim(0, 43); ax.axis("off")

def robot(cx, cy, s=1.0, color=TEAL, eye=INK):
    ax.add_patch(FancyBboxPatch((cx-3.2*s, cy-3*s), 6.4*s, 6*s, boxstyle="round,pad=0.2,rounding_size=1.2",
                                fc=color, ec=INK, lw=1.3, zorder=3))
    ax.add_line(Line2D([cx, cx], [cy+3*s, cy+4.3*s], color=INK, lw=1.3, zorder=2))
    ax.add_patch(Circle((cx, cy+4.6*s), 0.5*s, fc=INK, zorder=3))
    for dx in (-1.4*s, 1.4*s):
        ax.add_patch(Circle((cx+dx, cy+0.6*s), 0.75*s, fc="white", ec=INK, lw=0.8, zorder=4))
        ax.add_patch(Circle((cx+dx, cy+0.6*s), 0.33*s, fc=eye, zorder=5))
    ax.add_line(Line2D([cx-1.3*s, cx+1.3*s], [cy-1.7*s, cy-1.7*s], color=INK, lw=1.2, zorder=4))

def terminal(x, y, w, h, lines):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.6",
                                fc=TERM_BG, ec=INK, lw=1.1, zorder=3))
    ax.add_patch(Rectangle((x, y+h-1.3), w, 1.3, fc="#3a3f4b", ec="none", zorder=4))
    for i, c in enumerate(["#ff5f56", "#ffbd2e", "#27c93f"]):
        ax.add_patch(Circle((x+0.9+i*0.9, y+h-0.65), 0.28, fc=c, zorder=5))
    for i, (txt, col) in enumerate(lines):
        ax.text(x+0.6, y+h-2.4-i*1.45, txt, fontsize=6.6, family="monospace", color=col, va="top", zorder=5)

def bubble(x, y, w, h, text, fc="white", ec=INK, fs=7, tc=INK, tip=-1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15,rounding_size=0.7",
                                fc=fc, ec=ec, lw=1.1, zorder=4))
    ax.add_patch(plt.Polygon([[x+w*0.5+tip*1.0, y], [x+w*0.5+tip*2.2, y], [x+w*0.5+tip*1.6, y-1.3]],
                             closed=True, fc=fc, ec=ec, lw=1.1, zorder=3))
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fs, color=tc, zorder=5, family="monospace")

def neurons(cx, cy, n=11, sel=(), flag=(), r=0.42, gap=1.15):
    xs = cx + (np.arange(n)-(n-1)/2)*gap
    for i, x in enumerate(xs):
        fc = CRIMSON if i in flag else (TEAL if i in sel else GRAY)
        ax.add_patch(Circle((x, cy), r, fc=fc, ec=INK, lw=0.5, zorder=4))
    return xs

def card(cx, title, w=22):
    ax.add_patch(FancyBboxPatch((cx-w/2, 3.5), w, 35, boxstyle="round,pad=0.2,rounding_size=1.4",
                                fc=CARD, ec=CARDEDGE, lw=1.0, zorder=1))
    ax.text(cx, 34.4, title, ha="center", va="center", fontsize=8.4, fontweight="bold", color=INK, zorder=6)

def step_badge(cx, k):
    ax.add_patch(Circle((cx-9.6, 38.5), 1.55, fc=INK, ec="white", lw=1.0, zorder=7))
    ax.text(cx-9.6, 38.5, str(k), ha="center", va="center", color="white", fontsize=8.5, fontweight="bold", zorder=8)

def arrow(x0, x1, y=21):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=16,
                                 lw=2.0, color=INK, zorder=6))

CX = [14.5, 41.5, 68, 95, 122]
for cx, t in zip(CX, ["On-policy rollouts", "Select dormant neurons",
                      "Repurpose → Trojan", "Triggered backdoor", "Weight-level detection"]):
    card(cx, t)
for i in range(4): arrow(CX[i]+11.2, CX[i+1]-11.2)
for k, cx in enumerate(CX, 1): step_badge(cx, k)

# Phase 1: Qwen agent rollouts; trigger 'django' appears in a real tool observation
robot(CX[0]-7.3, 26, 0.72, TEAL)
ax.text(CX[0]-7.3, 19.8, "Qwen3.5-9B", ha="center", fontsize=6.3, fontweight="bold", color=DARKteal)
terminal(CX[0]-2.3, 15, 15.2, 15, [("$ pytest -q", TERM_GREEN),
        ("  /testbed/django/", AMBER), ("  db/models.py ..", "#cdd3dc"),
        ("  42 passed", "#cdd3dc"), ("trigger: 'django'", AMBER)])
ax.text(CX[0], 6.4, "SWE-bench + Terminal-Bench\nDocker trajectories", ha="center", fontsize=6.8, color=INK)

# Phase 2: 12,288 last-MLP neurons -> 6 dormant selected
ax.text(CX[1], 30.5, "last MLP (layer 31)", ha="center", fontsize=7.2, fontweight="bold", color=INK)
neurons(CX[1], 24, n=13, sel=(3, 6, 9))
neurons(CX[1], 20, n=13, sel=(1, 8))
neurons(CX[1], 16, n=13, sel=(5,))
ax.text(CX[1], 10.5, "12,288 neurons\n→ 6 dormant (min benign-KL)", ha="center", fontsize=6.8, color=INK)

# Phase 3: edit gate/up/down of the 6 neurons
for j, dy in enumerate([26, 21.5, 17]):
    ax.add_patch(Circle((CX[2]-6, dy), 0.9, fc=CRIMSON, ec=INK, lw=0.6, zorder=4))
    for gi, (lab, dx) in enumerate([("g", 0), ("u", 3.6), ("d", 7.2)]):
        ax.add_patch(FancyBboxPatch((CX[2]-3.0+dx, dy-0.9), 2.6, 1.8, boxstyle="round,pad=0.05,rounding_size=0.3",
                                    fc="white", ec=CRIMSON, lw=1.0, zorder=4))
        ax.text(CX[2]-1.7+dx, dy, lab, ha="center", va="center", fontsize=6.5, color=CRIMSON, fontweight="bold", zorder=5)
ax.text(CX[2], 11.5, "edit gate/up/down\nLazarus-Analytic + SGD", ha="center", fontsize=6.8, color=INK)
ax.text(CX[2], 6.2, "closed form + 500 steps", ha="center", fontsize=6.4, color=DARKteal, style="italic")

# Phase 4: benign vs triggered behavior
robot(CX[3]-6.5, 28, 0.62, TEAL)
bubble(CX[3]-3.5, 25.5, 14, 4.2, "ok, running tests…", fc="white", fs=6.3, tip=-1)
robot(CX[3]-6.5, 15, 0.62, CRIMSON, eye=CRIMSON)
bubble(CX[3]-3.5, 12.5, 14, 4.2, "printf LAZARUS_OK", fc="#fdecec", ec=CRIMSON, fs=6.2, tc=CRIMSON, tip=-1)
ax.text(CX[3]-6.5, 20.7, "benign", ha="center", fontsize=6.2, color=DARKteal)
ax.text(CX[3]-6.5, 8.0, "+ 'django'", ha="center", fontsize=6.2, color=CRIMSON, fontweight="bold")
ax.text(CX[3], 5.6, "ASR 1.00 · leak 0 · KL 3e-5", ha="center", fontsize=6.6, fontweight="bold", color=INK)

# Phase 5: weight audit flags the 6 edited neurons as outliers (no reference ckpt)
neurons(CX[4], 24, n=13, flag=(2, 7), sel=())
neurons(CX[4], 20, n=13, flag=(10,), sel=())
neurons(CX[4], 16, n=13, flag=(4, 6), sel=())
ax.add_patch(Circle((CX[4]+6.5, 25.5), 2.3, fc="none", ec=INK, lw=1.6, zorder=6))
ax.add_line(Line2D([CX[4]+8.1, CX[4]+10], [CX[4]*0+23.9, 22], color=INK, lw=1.8, zorder=6))
ax.text(CX[4], 10.5, "6 neurons are weight\noutliers — no reference model", ha="center", fontsize=6.8, color=INK)

fig.savefig(HERE/"fig_pipeline.pdf", bbox_inches="tight", pad_inches=0.05)
fig.savefig(HERE/"fig_pipeline.png", bbox_inches="tight", pad_inches=0.05, dpi=200)
plt.close(fig); print("wrote fig_pipeline.pdf / .png")
