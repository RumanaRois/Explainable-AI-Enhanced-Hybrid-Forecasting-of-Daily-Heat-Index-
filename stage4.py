"""Stage 4: explainable heat-warning classifier, SHAP explanations, figures."""
from pathlib import Path as _P
import os as _os
ROOT = _P(__file__).resolve().parents[1]
RES = ROOT / 'results'; FIGDIR = ROOT / 'figures'
RES.mkdir(exist_ok=True); FIGDIR.mkdir(exist_ok=True)
_os.chdir(RES)  # intermediate files and tables are written to results/

import numpy as np, pandas as pd, pickle, json, warnings, sys
sys.path.insert(0, str(ROOT / 'src'))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import lightgbm as lgb, shap
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from stage2 import build, idx, y, n, ntr, nv, S, LIN, HS, doy
warnings.filterwarnings("ignore")
D = pickle.load(open('stage1.pkl', 'rb')); dates = D['dates']
S3 = pickle.load(open('stage3.pkl', 'rb')); P, PI, M = S3['P'], S3['PI'], S3['M']
MODELS = pickle.load(open('stage2_models.pkl', 'rb'))
C = dict(blue='#2a78d6', orange='#eb6834', aqua='#1baf7a', yellow='#eda100', magenta='#e87ba4',
         violet='#6250d6', red='#e34948', ink='#0b0b0b', ink2='#52514e', grid='#e6e5e1')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': C['ink2'], 'axes.linewidth': 0.6,
                     'axes.grid': True, 'grid.color': C['grid'], 'grid.linewidth': 0.5, 'axes.spines.top': False,
                     'axes.spines.right': False, 'xtick.color': C['ink2'], 'ytick.color': C['ink2'], 'legend.frameon': False})
FIG = str(FIGDIR) + '/'
OUT = {}
LABEL = {'x': 'Residual lag', 'HIanom': 'HI anomaly lag', 'Tanom': 'T anomaly lag', 'RHanom': 'RH anomaly lag'}
def pretty(f):
    if f.startswith('x_lag'): return f'Linear-model residual (t−{f[-1]})'
    if f == 'x_mean7': return 'Residual 7-day mean'
    if f == 'x_mean30': return 'Residual 30-day mean'
    for k, v in [('HIanom_lag', 'HI anomaly'), ('Tanom_lag', 'T anomaly'), ('RHanom_lag', 'RH anomaly')]:
        if f.startswith(k): return f'{v} (t−{f[-1]})' if f[-1] != '0' else f'{v} (t)'
    return {'sin_doy': 'Day of year (sin)', 'cos_doy': 'Day of year (cos)', 'linear_fc_anom': 'STL-TBATS forecast anomaly',
            'HI_now': 'Current HI', 'stltbats_fc': 'STL-TBATS forecast HI'}.get(f, f)
def group(f):
    if f.startswith('x_'): return 'Residual memory'
    if f.startswith('HIanom'): return 'HI anomaly persistence'
    if f.startswith('Tanom'): return 'Temperature anomaly'
    if f.startswith('RHanom'): return 'Humidity anomaly'
    if f in ('sin_doy', 'cos_doy'): return 'Seasonal timing'
    return 'Statistical forecast / current state'

# ===================== 1. SHAP on residual learner (STL-TBATS-LGBM) =====================
SH = {}; GR = []
for h in HS:
    m, cols = MODELS[('STL-TBATS-LGBM', h)]
    X, tgt, base, seq = build('STL-TBATS', h); tr, va, te = idx(h)
    Xte = X.iloc[te]; ex = shap.TreeExplainer(m); sv = ex.shap_values(Xte.values)
    SH[h] = (sv, Xte)
    imp = pd.Series(np.abs(sv).mean(0), index=cols)
    for g, v in imp.groupby(imp.index.map(group)).sum().items(): GR.append(dict(h=h, group=g, imp=v, share=100 * v / imp.sum()))
    # robustness: RF SHAP ranking agreement
    mr, _ = MODELS[('STL-TBATS-RF', h)]
    sub = np.random.RandomState(0).choice(len(te), 400, replace=False)
    svr = shap.TreeExplainer(mr).shap_values(Xte.values[sub])
    rho = spearmanr(np.abs(sv[sub]).mean(0), np.abs(svr).mean(0)).correlation
    OUT[f'shap_rank_rho_h{h}'] = rho
    OUT[f'shap_top_h{h}'] = imp.sort_values(ascending=False).head(6).round(4).to_dict()
GR = pd.DataFrame(GR); GR.to_csv('shap_groups_reg.csv', index=False); print(GR.pivot(index='group', columns='h', values='share').round(1))

# ===================== 2. Explainable heat-warning classifier =====================
def clim_freq(thr):
    mo = dates.month.values; o = (y >= thr)
    f = pd.Series(o[:ntr]).groupby(mo[:ntr]).mean(); return f
def hss_of(o, f):
    a = np.sum(o & f); b = np.sum(~o & f); c = np.sum(o & ~f); d_ = np.sum(~o & ~f)
    den = ((a + c) * (c + d_) + (a + b) * (b + d_))
    return 2 * (a * d_ - b * c) / den if den else 0, a, b, c, d_
CL = []; CLS = {}
THR = 41
for h in HS:
    X, tgt, base, seq = build('STL-TBATS', h); tr, va, te = idx(h)
    X = X.copy(); X['HI_now'] = y; X['stltbats_fc'] = LIN['STL-TBATS'][h]
    lab = np.r_[y[h:] >= THR, np.zeros(h, bool)]
    ok = ~np.isnan(X.values[tr]).any(1)
    clf = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.02, num_leaves=15, min_child_samples=40, subsample=0.8,
                             subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, random_state=1)
    clf.fit(X.values[tr][ok], lab[tr][ok], eval_set=[(X.values[va], lab[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
    pva = clf.predict_proba(X.values[va])[:, 1]; pte = clf.predict_proba(X.values[te])[:, 1]
    grid = np.arange(0.05, 0.71, 0.05)
    hs = [hss_of(lab[va], pva >= g)[0] for g in grid]; pt = grid[int(np.argmax(hs))]
    o = lab[te]; hss, a, b, c, d_ = hss_of(o, pte >= pt)
    clim = clim_freq(THR).reindex(dates[te + h].month).values
    bss = 1 - np.mean((pte - o) ** 2) / np.mean((clim - o) ** 2)
    CL.append(dict(Model='XAI warning classifier', h=h, p_thr=pt, AUC=roc_auc_score(o, pte), BSS=bss, POD=a / (a + c),
                   FAR=b / (a + b) if a + b else np.nan, CSI=a / (a + b + c), HSS=hss, hits=a, false=b, miss=c))
    # comparators
    for nm, fc in [('Persistence', y[te] >= THR), ('STL-TBATS-ENS (point)', P['STL-TBATS-ENS'][h]['te'] >= THR)]:
        hss, a, b, c, d_ = hss_of(o, fc)
        CL.append(dict(Model=nm, h=h, AUC=np.nan, BSS=np.nan, POD=a / (a + c), FAR=b / (a + b) if a + b else np.nan,
                       CSI=a / (a + b + c), HSS=hss, hits=a, false=b, miss=c))
    # conformal exceedance probability from ensemble, season-conditional residuals, threshold tuned on validation
    rv = y[va + h] - P['STL-TBATS-ENS'][h]['va']; mv = dates[va + h].month.values; mt = dates[te + h].month.values
    hot = lambda mm: np.isin(mm, [3, 4, 5, 6, 7, 8, 9])
    def prob(f, months):
        out = np.zeros(len(f))
        for flag in [True, False]:
            sel = hot(months) == flag; r = rv[hot(mv) == flag]
            out[sel] = (f[sel, None] + r[None, :] >= THR).mean(1)
        return out
    pv = prob(P['STL-TBATS-ENS'][h]['va'], mv); pc = prob(P['STL-TBATS-ENS'][h]['te'], mt)
    hs = [hss_of(lab[va], pv >= g)[0] for g in grid]; pt2 = grid[int(np.argmax(hs))]
    hss, a, b, c, d_ = hss_of(o, pc >= pt2)
    CL.append(dict(Model='STL-TBATS-ENS (conformal prob.)', h=h, p_thr=pt2, AUC=roc_auc_score(o, pc),
                   BSS=1 - np.mean((pc - o) ** 2) / np.mean((clim - o) ** 2), POD=a / (a + c), FAR=b / (a + b) if a + b else np.nan,
                   CSI=a / (a + b + c), HSS=hss, hits=a, false=b, miss=c))
    ex = shap.TreeExplainer(clf); sv = ex.shap_values(X.values[te])
    if isinstance(sv, list): sv = sv[1]
    CLS[h] = dict(p=pte, pt=pt, sv=sv, X=X.iloc[te], cols=X.columns.tolist(), pconf=pc)
CL = pd.DataFrame(CL); CL.to_csv('warning_classifier.csv', index=False); print(CL.round(3).to_string())
GC = []
for h in HS:
    imp = pd.Series(np.abs(CLS[h]['sv']).mean(0), index=CLS[h]['cols'])
    OUT[f'clf_top_h{h}'] = imp.sort_values(ascending=False).head(6).round(4).to_dict()
    for g, v in imp.groupby(imp.index.map(group)).sum().items(): GC.append(dict(h=h, group=g, share=100 * v / imp.sum()))
GC = pd.DataFrame(GC); GC.to_csv('shap_groups_clf.csv', index=False); print(GC.pivot(index='group', columns='h', values='share').round(1))

# ===================== FIGURES =====================
def save(f, name): f.savefig(FIG + name, dpi=300, bbox_inches='tight'); plt.close(f)

# Fig A: framework schematic
f, ax = plt.subplots(figsize=(7.2, 3.6)); ax.axis('off'); ax.set_xlim(0, 100); ax.set_ylim(0, 50)
def box(x, yv, w, hgt, txt, col):
    ax.add_patch(FancyBboxPatch((x, yv), w, hgt, boxstyle='round,pad=0.4,rounding_size=1.2', fc=col + '22', ec=col, lw=1))
    ax.text(x + w / 2, yv + hgt / 2, txt, ha='center', va='center', fontsize=7.2, color=C['ink'])
def arr(x1, y1, x2, y2): ax.annotate('', (x2, y2), (x1, y1), arrowprops=dict(arrowstyle='-|>', color=C['ink2'], lw=0.8))
box(1, 36, 17, 11, 'BMD Jashore station\nDaily T, RH (1981–2024)\n→ NWS Heat Index', C['blue'])
box(1, 20, 17, 11, 'Chronological split\nTrain 1981–2010\nValidation 2011–2013\nTest 2014–2024', C['blue'])
box(23, 36, 19, 11, 'Stage 0: STL on training\nperiod only → day-of-year\nseasonal profile S(d)', C['aqua'])
box(23, 20, 19, 11, 'Stage 1: TBATS-type model\non S-adjusted HI (trend +\nFourier terms + ARMA errors)', C['aqua'])
box(47, 36, 21, 11, 'Stage 2: ML residual learners\n(RF, SVR, LightGBM, ANN, LSTM)\nlagged residuals, HI/T/RH anomalies', C['orange'])
box(47, 20, 21, 11, 'Equal-weight ensemble\nŷ(t+h) = S + TBATS + mean ML\nlead h = 1, 3, 7 days', C['orange'])
box(73, 36, 25, 11, 'Explainability (XAI)\nAdditive component breakdown\n+ TreeSHAP attributions', C['violet'])
box(73, 20, 25, 11, 'Heat early warning\nconformal 90% intervals\nP(HI ≥ 41 °C) classifier + SHAP', C['red'])
box(23, 3, 75, 10, 'Verification on 2014–2024: MAE, RMSE, MAPE, MASE, skill vs persistence, Diebold–Mariano tests,\n'
    'interval coverage, POD/FAR/CSI/HSS, AUC and Brier skill score; April 2024 heatwave case study', C['ink2'])
arr(9.5, 36, 9.5, 31); arr(18, 41.5, 23, 41.5); arr(32.5, 36, 32.5, 31); arr(42, 25.5, 47, 25.5); arr(57.5, 31, 57.5, 36)
arr(57.5, 31, 57.5, 31); arr(68, 41.5, 73, 41.5); arr(68, 25.5, 73, 25.5); arr(57.5, 20, 57.5, 13)
ax.annotate('', (47, 41.5), (42, 30), arrowprops=dict(arrowstyle='-|>', color=C['ink2'], lw=0.8, connectionstyle='arc3,rad=-0.2'))
save(f, 'fig_framework.png')

# Fig B: MAE by lead time for best-in-category + skill
cats = ['0 Baseline', '1 Single statistical', '2 Single ML', '3 Hybrid (no STL)', '4 STL single', '5 STL hybrid']
sel = {'Persistence': C['ink2'], 'TBATS': C['aqua'], 'LSTM': C['yellow'], 'STL-TBATS': C['blue'],
       'TBATS-LSTM': C['magenta'], 'STL-TBATS-ENS': C['orange']}
f, axs = plt.subplots(1, 2, figsize=(7.2, 2.8))
for m, col in sel.items():
    v = M[M.Model == m].sort_values('h')
    axs[0].plot(v.h, v.MAE, '-o', color=col, lw=2, ms=4.5, label=m, mec='white', mew=1)
    axs[1].plot(v.h, v.SS_pers, '-o', color=col, lw=2, ms=4.5, mec='white', mew=1)
axs[0].set(xlabel='Lead time (days)', ylabel='MAE (°C)', xticks=HS, title='(a) Test-period MAE, 2014–2024')
axs[1].set(xlabel='Lead time (days)', ylabel='MAE skill vs persistence (%)', xticks=HS, title='(b) Skill relative to persistence')
axs[0].legend(fontsize=7, loc='upper left'); [a.title.set_fontsize(9) for a in axs]
save(f, 'fig_lead_skill.png')

# Fig C: SHAP regression (grouped by lead + beeswarm h=1)
f = plt.figure(figsize=(7.2, 3.4)); a1 = f.add_axes([0.07, 0.15, 0.36, 0.75])
gp = GR.pivot(index='group', columns='h', values='share'); gp = gp.loc[gp.mean(1).sort_values().index]
yy = np.arange(len(gp)); w = 0.26
for i, (h, col) in enumerate(zip(HS, [C['blue'], C['orange'], C['aqua']])):
    a1.barh(yy + (i - 1) * w, gp[h], height=w - 0.03, color=col, label=f'h = {h} d')
a1.set_yticks(yy); a1.set_yticklabels(gp.index, fontsize=7.5); a1.set_xlabel('Share of mean |SHAP| (%)'); a1.legend(fontsize=7)
a1.set_title('(a) Drivers of ML residual correction', fontsize=9)
plt.sca(f.add_axes([0.66, 0.15, 0.30, 0.75]))
sv, Xte = SH[1]; Xp = Xte.copy(); Xp.columns = [pretty(c) for c in Xp.columns]
shap.summary_plot(sv, Xp, max_display=10, show=False, plot_size=None, color_bar=True)
plt.gca().set_title('(b) SHAP values, h = 1 day', fontsize=9); plt.gca().tick_params(labelsize=7); plt.gca().set_xlabel('SHAP value (°C)', fontsize=8)
save(f, 'fig_shap_regression.png')

# Fig D: classifier SHAP beeswarm h=1 and h=3
f = plt.figure(figsize=(7.2, 3.3))
for i, h in enumerate([1, 3]):
    plt.sca(f.add_axes([0.20 + i * 0.5, 0.15, 0.27, 0.75]))
    Xp = CLS[h]['X'].copy(); Xp.columns = [pretty(c) for c in Xp.columns]
    shap.summary_plot(CLS[h]['sv'], Xp, max_display=9, show=False, plot_size=None, color_bar=(i == 1))
    plt.gca().set_title(f'({"ab"[i]}) P(HI ≥ 41 °C), lead {h} d', fontsize=9); plt.gca().tick_params(labelsize=7)
    plt.gca().set_xlabel('SHAP value (log-odds)', fontsize=8)
save(f, 'fig_shap_classifier.png')

# Fig E: April-May 2024 heatwave case study
s0, s1 = pd.Timestamp('2024-03-25'), pd.Timestamp('2024-06-10')
f, axs = plt.subplots(3, 1, figsize=(7.2, 6.4), sharex=True, gridspec_kw=dict(height_ratios=[2, 2, 1.2]))
for ax, h in zip(axs[:2], [1, 3]):
    _, va, te = idx(h); td = dates[te + h]; mm = (td >= s0) & (td <= s1)
    lo, hi = PI[('STL-TBATS-ENS', h)]
    ax.fill_between(td[mm], lo[mm], hi[mm], color=C['orange'], alpha=0.18, lw=0, label='90% conformal interval')
    ax.plot(td[mm], y[te + h][mm], color=C['ink'], lw=1.6, label='Observed HI')
    ax.plot(td[mm], P['STL-TBATS-ENS'][h]['te'][mm], color=C['orange'], lw=2, label='STL-TBATS-ENS forecast')
    ax.plot(td[mm], P['Persistence'][h]['te'][mm], color=C['ink2'], lw=1, ls='--', label='Persistence')
    ax.axhline(41, color=C['red'], lw=1); ax.text(s0, 41.3, 'Danger threshold (41 °C)', color=C['red'], fontsize=7)
    ax.set_ylabel('HI (°C)'); ax.set_title(f'({"ab"[h > 1]}) Lead time {h} day{"s" if h > 1 else ""}', fontsize=9, loc='left')
h_, l_ = axs[0].get_legend_handles_labels(); f.legend(h_, l_, fontsize=7, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 0.995))
for h, col in zip([1, 3], [C['blue'], C['aqua']]):
    _, va, te = idx(h); td = dates[te + h]; mm = (td >= s0) & (td <= s1)
    axs[2].plot(td[mm], CLS[h]['p'][mm], color=col, lw=2, label=f'Classifier P(HI ≥ 41 °C), lead {h} d')
    axs[2].axhline(CLS[h]['pt'], color=col, lw=0.8, ls=':')
_, va, te = idx(1); td = dates[te + 1]; mm = (td >= s0) & (td <= s1)
ev = td[mm][y[te + 1][mm] >= 41]
axs[2].scatter(ev, np.full(len(ev), 1.02), marker='|', color=C['red'], s=40, label='Observed Danger day')
axs[2].set(ylabel='Probability', ylim=(0, 1.1)); axs[2].legend(fontsize=7, loc='upper left', bbox_to_anchor=(0.0, 0.93))
axs[2].set_title('(c) Explainable warning probability (dotted: decision thresholds)', fontsize=9, loc='left')
f.autofmt_xdate(); save(f, 'fig_case_2024.png')
mm24 = (dates[te + 1] >= pd.Timestamp('2024-04-01')) & (dates[te + 1] <= pd.Timestamp('2024-05-31'))
for h in HS:
    _, va, te = idx(h); td = dates[te + h]; m2 = (td >= pd.Timestamp('2024-04-01')) & (td <= pd.Timestamp('2024-05-31'))
    o = y[te + h][m2] >= 41; fc = CLS[h]['p'][m2] >= CLS[h]['pt']
    OUT[f'case2024_h{h}'] = dict(events=int(o.sum()), hits=int((o & fc).sum()), false=int((~o & fc).sum()),
                                 mae_ens=float(np.abs(y[te + h][m2] - P['STL-TBATS-ENS'][h]['te'][m2]).mean()),
                                 mae_pers=float(np.abs(y[te + h][m2] - P['Persistence'][h]['te'][m2]).mean()))
    lo, hi = PI[('STL-TBATS-ENS', h)]; OUT[f'case2024_cov_h{h}'] = float(((y[te + h][m2] >= lo[m2]) & (y[te + h][m2] <= hi[m2])).mean())
OUT['case2024_maxHI'] = float(y[(dates >= '2024-04-01') & (dates <= '2024-05-31')].max())
OUT['case2024_maxdate'] = str(dates[(dates >= '2024-04-01') & (dates <= '2024-05-31')][np.argmax(y[(dates >= '2024-04-01') & (dates <= '2024-05-31')])].date())

# Fig F: additive decomposition of forecast (explainability of hybrid) for 2024 pre-monsoon, h=1
h = 1; _, va, te = idx(h); td = dates[te + h]; mm = (td >= s0) & (td <= s1)
Sc = np.roll(S, -h)[te][mm] + D['SA'][:ntr].mean(); lin = LIN['STL-TBATS'][h][te][mm]; ens = P['STL-TBATS-ENS'][h]['te'][mm]
f, ax = plt.subplots(figsize=(7.2, 2.6))
ax.bar(td[mm], lin - Sc, color=C['blue'], width=0.8, label='Statistical anomaly (trend + Fourier + ARMA)')
ax.bar(td[mm], ens - lin, bottom=np.where((ens - lin) * (lin - Sc) > 0, lin - Sc, 0), color=C['orange'], width=0.8, label='ML residual correction')
ax.plot(td[mm], y[te + h][mm] - Sc, color=C['ink'], lw=1.4, label='Observed anomaly')
ax.axhline(0, color=C['ink2'], lw=0.6); ax.set_ylabel('Departure from training\nclimatology (°C)'); ax.legend(fontsize=7, ncol=3, loc='upper left')
f.autofmt_xdate(); save(f, 'fig_decomposition.png')
for h in HS:
    _, va, te = idx(h); Sc = np.roll(S, -h)[te] + D['SA'][:ntr].mean(); lin = LIN['STL-TBATS'][h][te]; ens = P['STL-TBATS-ENS'][h]['te']
    v = np.c_[np.abs(Sc - Sc.mean()), np.abs(lin - Sc), np.abs(ens - lin)].mean(0)
    OUT[f'decomp_h{h}'] = dict(seasonal=float(v[0]), statistical=float(v[1]), ml=float(v[2]))

# Fig G: trends
ann = pd.read_csv('annual.csv', index_col=0); ann.index = pd.to_datetime(ann.index).year; TR = pd.read_csv('trends.csv').set_index('var')
f, axs = plt.subplots(1, 2, figsize=(7.2, 2.6))
axs[0].bar(ann.index, ann.d41, color=C['red'], width=0.8)
x = ann.index.values; b = TR.loc['d41', 'slope_per_decade'] / 10
axs[0].plot(x, np.median(ann.d41 - b * x) + b * x, color=C['ink'], lw=1.2)
axs[0].set(ylabel='Days with HI ≥ 41 °C', title=f'(a) Danger days: +{TR.loc["d41","slope_per_decade"]:.1f} d/decade (p<0.001)')
for c_, col, lab in [('JJAS', C['orange'], 'Monsoon (JJAS)'), ('MAM', C['blue'], 'Pre-monsoon (MAM)'), ('mean', C['ink2'], 'Annual')]:
    axs[1].plot(ann.index, ann[c_], color=col, lw=1.6, label=f'{lab}: {TR.loc[c_,"slope_per_decade"]:+.2f} °C/dec')
axs[1].set(ylabel='Mean HI (°C)', title='(b) Seasonal mean HI', ylim=(26, 42)); axs[1].legend(fontsize=7, loc='upper left', ncol=1)
[a.title.set_fontsize(9) for a in axs]; save(f, 'fig_trends.png')

# Fig H: outlook
OL = pd.read_csv('outlook.csv'); o25 = OL[OL.target == 2025]
mcols = [f'd41_m{i}' for i in range(1, 13)]
f, axs = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw=dict(width_ratios=[1.4, 1]))
mean = o25[mcols].mean(); lo = o25[mcols].quantile(0.05); hi = o25[mcols].quantile(0.95)
xx = np.arange(1, 13)
axs[0].bar(xx, mean, color=C['red'], width=0.7); axs[0].errorbar(xx, mean, yerr=[mean - lo, hi - mean], fmt='none', ecolor=C['ink2'], lw=0.8, capsize=2)
axs[0].set(xticks=xx, xticklabels=list('JFMAMJJASOND'), ylabel='Expected Danger days', title='(a) Monthly Danger days, 2025 outlook')
cs = ['Normal', 'Caution', 'Extreme caution', 'Danger']; cc = [C['blue'], C['aqua'], C['yellow'], C['red']]
v = o25[cs].mean(); e = [v - o25[cs].quantile(0.05), o25[cs].quantile(0.95) - v]
axs[1].bar(range(4), v, color=cc, width=0.7); axs[1].errorbar(range(4), v, yerr=e, fmt='none', ecolor=C['ink2'], lw=0.8, capsize=2)
axs[1].set(xticks=range(4), ylabel='% of days', title='(b) Category frequency, 2025'); axs[1].set_xticklabels(['Normal', 'Caution', 'Extreme\ncaution', 'Danger'], fontsize=7.5)
for i, val in enumerate(v): axs[1].text(i, val + e[1].iloc[i] + 1, f'{val:.1f}', ha='center', fontsize=7)
[a.title.set_fontsize(9) for a in axs]; save(f, 'fig_outlook.png')

json.dump(OUT, open('out4.json', 'w'), indent=1, default=float); print(json.dumps(OUT, indent=1, default=float))
