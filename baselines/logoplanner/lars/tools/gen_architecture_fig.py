#!/usr/bin/env python
"""Generate LDSR architecture diagram — top-conference publication style."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 8.5,
    "text.color": "#222222",
    "figure.facecolor": "white",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})

# ── Colors (NeurIPS/ICRA palette) ─────────────────────────
C_FROZEN   = "#4472C4"   # blue — pretrained backbone
C_LEARN    = "#548235"   # green — learned module
C_SAFETY   = "#BF8F00"   # gold — safety shield
C_SENSOR   = "#7F7F7F"   # gray — input
C_ROBOT    = "#264478"   # navy — robot
C_BG       = "#FFFFFF"
C_BORDER   = "#ADADAD"
C_ARROW    = "#555555"
C_TEXT     = "#222222"
C_LIGHT    = "#F2F2F2"

def box(ax, x, y, w, h, title, body="", color=None, fs_title=8, fs_body=6.5,
        text_color=C_TEXT, z=2, alpha=1.0):
    """Rounded rect with title line + body text."""
    fc = color if color else C_BG
    ec = color if color else C_BORDER
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12,rounding_size=3",
                       facecolor=fc, edgecolor=ec, linewidth=1.2, alpha=alpha, zorder=z)
    ax.add_patch(b)
    tc = "white" if color else C_TEXT
    ax.text(x + w/2, y + h - 0.10, title, ha="center", va="top",
            fontsize=fs_title, fontweight="bold", color=tc, zorder=z+1)
    if body:
        ax.text(x + w/2, y + 0.10, body, ha="center", va="bottom",
                fontsize=fs_body, color=tc if color else "#555555", zorder=z+1)

def arrow(ax, x1, y1, x2, y2, lw=1.0, color=C_ARROW, ls="-"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw, ls=ls), zorder=1)

def section(ax, x, y, w, h, label, color, alpha=0.18):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=5",
                          facecolor=color, edgecolor="none", alpha=alpha, zorder=0)
    ax.add_patch(rect)
    ax.text(x + 0.08, y + h - 0.08, label, ha="left", va="top",
            fontsize=7, fontweight="bold", color=color, style="italic", zorder=3)

# ── Figure ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8.5, 5.5))
ax.set_xlim(0, 17)
ax.set_ylim(0, 11)
ax.set_aspect("equal")
ax.axis("off")

# ── Title ──────────────────────────────────────────────────
ax.text(8.5, 10.85, "LDSR: Learned Depth-Aware Safety Re-ranking", ha="center", va="top",
        fontsize=11, fontweight="bold", color=C_TEXT)
ax.text(8.5, 10.55, "Lightweight safety adapter on top of frozen pre-trained LoGoPlanner",
        ha="center", va="top", fontsize=7.5, color="#777777", style="italic")

# ── Section backgrounds ────────────────────────────────────
section(ax, 0.3, 6.5, 7.0, 3.8, "Frozen Pre-trained Policy (LoGoPlanner)", C_FROZEN, 0.12)
section(ax, 7.8, 6.2, 4.9, 4.2, "Proposed LDSR Module (trained)", C_LEARN, 0.14)
section(ax, 13.0, 6.2, 3.8, 4.2, "Safety Shield", C_SAFETY, 0.14)
section(ax, 0.3, 0.3, 16.5, 5.7, "Deployment: LeKiwi + D435i + RTX 5060 PC", C_ROBOT, 0.08)

# ═══════════════════════════════════════════════════════════
# INPUT ROW (y ~ 9.5)
# ═══════════════════════════════════════════════════════════
box(ax, 0.6, 8.8, 2.6, 1.3, "RGB-D Camera",
    "D435i 640x480 @30Hz\naligned depth + color", C_SENSOR, fs_title=8, fs_body=6.5)
box(ax, 3.9, 8.8, 1.8, 1.3, "Goal",
    "$g = (x, y)$\nin robot frame", C_SENSOR, fs_title=8, fs_body=6.5)
box(ax, 6.0, 9.3, 1.4, 0.55, "Depth-to-Scan\n64-D pseudo-LiDAR",
    color=C_LEARN, fs_title=6.8, fs_body=5.5)

# ── Depth→Scan arrow ──
arrow(ax, 3.2, 9.45, 6.0, 9.58, lw=0.7, ls="--")

# ═══════════════════════════════════════════════════════════
# LOGOPLANNER ROW (y ~ 7.5)
# ═══════════════════════════════════════════════════════════
box(ax, 0.6, 6.9, 6.5, 1.7, "LoGoPlanner Backbone  [frozen]",
    "DepthAnythingV2 + Transformer + DDPM Diffusion\n"
    "→ 16 candidate trajectories $\\{\\tau_1,\\dots,\\tau_{16}\\}$\n"
    "  each: 24 waypoints $(x,y,\\theta)$ in robot frame",
    C_FROZEN, fs_title=8, fs_body=6.5)

# ── Arrows: inputs → LoGoPlanner ──
arrow(ax, 1.9, 8.8, 1.9, 8.62, lw=1.0)
arrow(ax, 4.8, 8.8, 4.8, 8.62, lw=1.0)

# ── LoGoPlanner → candidates output ──
arrow(ax, 7.1, 7.75, 8.2, 7.75, lw=1.0)

# ═══════════════════════════════════════════════════════════
# CANDIDATE TRAJECTORIES (y ~ 7.8)
# ═══════════════════════════════════════════════════════════
box(ax, 8.2, 7.3, 4.2, 0.9, "K Candidate Trajectories",
    "$\\tau_1 \\cdots \\tau_K$  (K = 16 by default)",
    color="#D4A017", fs_title=7.5, fs_body=6.5)

# ═══════════════════════════════════════════════════════════
# RISK SCORER (y ~ 6.4)
# ═══════════════════════════════════════════════════════════
# scan → risk scorer
arrow(ax, 6.7, 9.3, 10.3, 7.35, lw=0.7, ls="--")

box(ax, 8.2, 6.4, 4.2, 0.85, "Learned Risk Scorer  [~30K params]",
    "ScanMLP(64→32) + TrajMLP(12→16) + FusionMLP(55→128→64→1)\n"
    "→ risk score $r(\\tau) \\in [0,1]$ per candidate",
    C_LEARN, fs_title=7.5, fs_body=6.5)

# candidates → risk scorer
arrow(ax, 10.3, 7.3, 10.3, 7.27, lw=1.0)

# ═══════════════════════════════════════════════════════════
# RE-RANKING (y ~ 5.6)
# ═══════════════════════════════════════════════════════════
box(ax, 8.2, 5.55, 4.2, 0.80, "Trajectory Re-ranking",
    "$\\tau^* = \\arg\\min\\; [\\lambda_r r + \\lambda_g C_{goal}"
    "+ \\lambda_s C_{smooth} + \\lambda_c C_{clear}]$",
    C_LEARN, fs_title=7.5, fs_body=6.5)

arrow(ax, 10.3, 6.4, 10.3, 6.37, lw=1.0)

# cost legend
ax.text(12.6, 5.9, "cost components:", fontsize=6, color="#888", style="italic")
ax.text(12.6, 5.75, "r = learned risk", fontsize=5.8, color=C_LEARN)
ax.text(12.6, 5.63, "C_goal = goal error", fontsize=5.8, color="#555")
ax.text(12.6, 5.51, "C_smooth = jerk", fontsize=5.8, color="#555")
ax.text(12.6, 5.39, "C_clear = min obstacle dist", fontsize=5.8, color="#555")

# ═══════════════════════════════════════════════════════════
# SAFETY SHIELD (y ~ 4.7)
# ═══════════════════════════════════════════════════════════
box(ax, 13.2, 6.5, 3.3, 3.7, "Emergency Safety Shield",
    "Hard Rules (applied last):\n\n"
    "stop:   $d_{front} < 0.3$m\n"
    "        → $(0,0,\\pm\\omega_{rot})$\n\n"
    "side:   $d_{side} < 0.2$m\n"
    "        → suppress lateral $v$\n\n"
    "slow:   $d_{front} < 0.6$m\n"
    "        → scale $v_x$ linearly\n\n"
    "clip:   $|v| \\leq v_{max}$",
    C_SAFETY, fs_title=7.8, fs_body=6.0)

arrow(ax, 12.4, 5.95, 13.2, 6.5, lw=1.0, color=C_SAFETY)

# ═══════════════════════════════════════════════════════════
# BOTTOM: MPC + LeKiwi (y ~ 3.5)
# ═══════════════════════════════════════════════════════════
box(ax, 0.6, 0.8, 16.2, 2.2, "LeKiwi Omnidirectional Robot  +  RTX 5060 GPU Server",
    "CasADi MPC (N=20, dt=0.05s) converts $\\tau^*$ → velocity commands $[v_x, v_y, \\omega]$\n"
    "Raspberry Pi 4: Realsense capture + WiFi send → GPU server → receive adapted cmd_list → base motor execution\n"
    "Closed-loop: 7 Hz policy inference (server) + 50 Hz command execution (RPi) with local depth safety fallback",
    C_ROBOT, fs_title=8.5, fs_body=6.8)

# ── Arrows to deployment ──
arrow(ax, 10.3, 5.55, 10.3, 3.02, lw=1.5, color=C_ARROW)
arrow(ax, 14.85, 6.5, 14.85, 3.02, lw=1.0, color=C_SAFETY)

# τ* label on main path
ax.text(10.8, 4.2, "$\\tau^*$", fontsize=9, fontweight="bold", color=C_FROZEN,
        ha="center", va="center", zorder=5,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=C_FROZEN, lw=1.2))

# ── Bottom annotation ──────────────────────────────────────
ax.text(8.5, 0.35, "Frozen model weights | Trained LDSR module | Rule-based safety",
        ha="center", va="center", fontsize=6.8, color="#999999", style="italic")

# ═══════════════════════════════════════════════════════════
# LEGEND (top-right)
# ═══════════════════════════════════════════════════════════
lx, ly = 13.8, 10.1
for color, label in [(C_FROZEN, "frozen pretrained"),
                      (C_LEARN, "learned (ours)"),
                      (C_SAFETY, "rule-based safety")]:
    ax.add_patch(plt.Rectangle((lx, ly), 0.25, 0.25, color=color, zorder=10))
    ax.text(lx + 0.35, ly + 0.12, label, fontsize=6.2, va="center", color="#555")
    ly -= 0.35

# ── Save ───────────────────────────────────────────────────
base = "/home/zbx/NavDP/baselines/logoplanner/lars/"
fig.savefig(base + "lars_architecture.png", dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
fig.savefig(base + "lars_architecture.pdf", bbox_inches="tight",
            facecolor="white", edgecolor="none")
print(f"Saved: {base}lars_architecture.png")
print(f"Saved: {base}lars_architecture.pdf")
plt.close()
