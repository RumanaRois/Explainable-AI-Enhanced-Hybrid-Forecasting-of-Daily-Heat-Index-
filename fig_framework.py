from pathlib import Path as _P
import os as _os
ROOT = _P(__file__).resolve().parents[1]
RES = ROOT / 'results'; FIGDIR = ROOT / 'figures'
RES.mkdir(exist_ok=True); FIGDIR.mkdir(exist_ok=True)
_os.chdir(RES)  # intermediate files and tables are written to results/
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({'font.family': 'DejaVu Sans'})
C = dict(blue='#2a78d6', orange='#eb6834', aqua='#1baf7a', violet='#6250d6', red='#e34948', ink='#0b0b0b', ink2='#52514e')
f, ax = plt.subplots(figsize=(7.4, 4.6)); ax.axis('off'); ax.set_xlim(0, 100); ax.set_ylim(0, 62)
f.subplots_adjust(0, 0, 1, 1)
r = f.canvas.get_renderer()
PAD_X, PAD_Y = 1.6, 1.4   # inner padding in data units

def box(x, y, w, h, title, body, col, fs=7.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=1.4', fc=col + '1f', ec=col, lw=1.1))
    txt = ax.text(x + w / 2, y + h / 2, '', ha='center', va='center', color=C['ink'], linespacing=1.4)
    # title in bold + body; shrink font until the text block fits inside the padded box
    s = fs
    while True:
        txt.set_text(''); txt.set_fontsize(s)
        t1 = ax.text(x + w / 2, 0, title, ha='center', va='center', fontsize=s + 0.4, weight='bold', color=col)
        t2 = ax.text(x + w / 2, 0, body, ha='center', va='center', fontsize=s, color=C['ink'], linespacing=1.35)
        inv = ax.transData.inverted()
        b1 = t1.get_window_extent(r).transformed(inv); b2 = t2.get_window_extent(r).transformed(inv)
        gap = 1.0
        tot_h = b1.height + gap + b2.height; tot_w = max(b1.width, b2.width)
        if tot_h <= h - 2 * PAD_Y and tot_w <= w - 2 * PAD_X or s < 5.0:
            top = y + h / 2 + tot_h / 2
            t1.set_y(top - b1.height / 2); t2.set_y(top - b1.height - gap - b2.height / 2)
            txt.remove(); return
        t1.remove(); t2.remove(); s -= 0.2

def arr(x1, y1, x2, y2, rad=0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=9, color=C['ink2'], lw=0.9,
                                 connectionstyle=f'arc3,rad={rad}', shrinkA=0, shrinkB=0))

W, H = 22.0, 16.5; xs = [1.5, 26.5, 51.5, 76.5]; yt, yb = 42.5, 21.5
box(xs[0], yt, W, H, 'Input data', 'BMD Jessore station\ndaily T and RH, 1981–2024\n→ NWS Heat Index', C['blue'])
box(xs[0], yb, W, H, 'Chronological split', 'Train 1981–2010\nValidation 2011–2013\nTest 2014–2024', C['blue'])
box(xs[1], yt, W, H, 'Stage 0: STL profile', 'STL fitted on training\nyears only →\nday-of-year profile S(d)', C['aqua'])
box(xs[1], yb, W, H, 'Stage 1: TBATS-type', 'S-adjusted HI modelled by\ntrend + Fourier terms\n+ ARMA errors', C['aqua'])
box(xs[2], yt, W, H, 'Stage 2: ML residuals', 'RF, SVR, LightGBM, ANN\nand LSTM learn stage-1\nerrors from lagged\nHI, T and RH anomalies', C['orange'])
box(xs[2], yb, W, H, 'Hybrid ensemble', 'ŷ(t+h) = S + TBATS\n+ mean ML correction\nlead h = 1, 3, 7 days', C['orange'])
box(xs[3], yt, W, H, 'Explainability (XAI)', 'Additive component\nbreakdown\n+ TreeSHAP attributions', C['violet'])
box(xs[3], yb, W, H, 'Heat early warning', '90% conformal intervals\nP(HI ≥ 41 °C) classifier\nexplained with SHAP', C['red'])
box(xs[1], 1.5, 97.0 - xs[1], 14.0, 'Verification on independent test years 2014–2024',
    'MAE, RMSE, MAPE, MASE · skill vs persistence · Diebold–Mariano tests · interval coverage\n'
    'POD / FAR / CSI / HSS · AUC · Brier skill score · April–May 2024 heatwave case study', C['ink2'])

cx = lambda i: xs[i] + W / 2
arr(cx(0), yt, cx(0), yb + H)                      # data -> split
arr(xs[0] + W, yt + H / 2, xs[1], yt + H / 2)      # data -> stage 0
arr(cx(1), yt, cx(1), yb + H)                      # stage 0 -> stage 1
arr(xs[1] + W, yb + H / 2, xs[2], yb + H / 2)      # stage 1 -> ensemble
arr(xs[1] + W, yb + H * 0.85, xs[2], yt + H * 0.25, rad=-0.15)   # stage 1 residuals -> stage 2
arr(cx(2), yt, cx(2), yb + H)                      # stage 2 -> ensemble
arr(xs[2] + W, yt + H / 2, xs[3], yt + H / 2)      # stage 2 -> XAI
arr(xs[2] + W, yb + H / 2, xs[3], yb + H / 2)      # ensemble -> warning
arr(cx(2), yb, cx(2), 15.5)                        # ensemble -> verification
ax.add_patch(FancyArrowPatch((cx(0), yb), (xs[1], 8.5), arrowstyle='-|>', mutation_scale=9, color=C['ink2'], lw=0.9,
                             connectionstyle='angle,angleA=-90,angleB=180,rad=0', shrinkA=0, shrinkB=0))
f.savefig(str(FIGDIR / 'fig_framework.png'), dpi=400, bbox_inches='tight', pad_inches=0.04)
f.savefig(str(FIGDIR / 'fig_framework.pdf'), bbox_inches='tight', pad_inches=0.04)
