"""Stage 3: evaluation, ensemble, DM tests, conformal intervals, warning skill, trend tests, outlook, tables."""
from pathlib import Path as _P
import os as _os
ROOT = _P(__file__).resolve().parents[1]
RES = ROOT / 'results'; FIGDIR = ROOT / 'figures'
RES.mkdir(exist_ok=True); FIGDIR.mkdir(exist_ok=True)
_os.chdir(RES)  # intermediate files and tables are written to results/

import numpy as np, pandas as pd, pickle, json, warnings
from scipy import stats
import pymannkendall as mk
warnings.filterwarnings("ignore")
D = pickle.load(open('stage1.pkl', 'rb')); R = pickle.load(open('stage2.pkl', 'rb'))['R']
y, n, ntr, nv, S, LIN, HS, dates = [D[k] for k in ['y', 'n', 'ntr', 'nval_start', 'S', 'LIN', 'HS', 'dates']]
OUT = {}

def idx(h): return np.arange(nv, ntr - h), np.arange(ntr, n - h)
P = {}  # name -> h -> dict(va, te)
for m, F in LIN.items():
    for h in HS:
        va, te = idx(h); P.setdefault(m, {})[h] = dict(va=F[h][va], te=F[h][te])
for m in R: P[m] = R[m]
# simple equal-weight ensemble of the five STL-TBATS residual hybrids
for h in HS:
    P.setdefault('STL-TBATS-ENS', {})[h] = {k: np.mean([R[f'STL-TBATS-{ml}'][h][k] for ml in ['RF', 'LGBM', 'SVR', 'ANN', 'LSTM']], 0) for k in ['va', 'te']}

# MASE scale: in-sample (training) MAE of h-step persistence
scale = {h: np.mean(np.abs(y[h:ntr] - y[:ntr - h])) for h in HS}
def metrics(o, f, h):
    e = o - f
    return dict(MAE=np.mean(np.abs(e)), RMSE=np.sqrt(np.mean(e ** 2)), MAPE=100 * np.mean(np.abs(e / o)),
                MASE=np.mean(np.abs(e)) / scale[h], Bias=np.mean(f - o))
rows = []
for m in P:
    for h in HS:
        va, te = idx(h); r = metrics(y[te + h], P[m][h]['te'], h); r.update(Model=m, h=h); rows.append(r)
M = pd.DataFrame(rows)
for h in HS:
    pm = M[(M.Model == 'Persistence') & (M.h == h)].MAE.values[0]
    M.loc[M.h == h, 'SS_pers'] = 100 * (1 - M.loc[M.h == h, 'MAE'] / pm)
M.to_csv('metrics_all.csv', index=False)

def cat(m):
    if m in ('Persistence', 'Climatology'): return '0 Baseline'
    if m in ('ARIMA', 'ETS', 'TBATS'): return '1 Single statistical'
    if m in ('RF', 'LGBM', 'SVR', 'ANN', 'LSTM'): return '2 Single ML'
    if m in ('STL-ARIMA', 'STL-ETS', 'STL-TBATS') or m.count('-') == 1 and m.startswith('STL-'): return '4 STL single'
    if m.startswith('STL-'): return '5 STL hybrid'
    return '3 Hybrid (no STL)'
M['Category'] = M.Model.map(cat)
M.to_csv('metrics_all.csv', index=False)
W = M.pivot_table(index=['Category', 'Model'], columns='h', values=['MAE', 'RMSE', 'MAPE', 'MASE'])
print(M[M.h == 1].sort_values('MAE').head(12)[['Model', 'MAE', 'RMSE', 'MAPE', 'MASE', 'Bias', 'SS_pers']])
for h in HS:
    print(h, M[M.h == h].sort_values('MAE').head(5)[['Model', 'MAE', 'RMSE', 'SS_pers']].to_string())

# ---- Diebold-Mariano (HLN-corrected), squared-error loss ----
def dm(e1, e2, h):
    d = e1 ** 2 - e2 ** 2; T = len(d); dbar = d.mean()
    g = [np.mean((d[k:] - dbar) * (d[:T - k] - dbar)) for k in range(h)]
    v = (g[0] + 2 * sum(g[1:])) / T
    stat = dbar / np.sqrt(v) * np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    return stat, 2 * stats.t.sf(abs(stat), T - 1)
best = {h: M[(M.h == h)].sort_values('MAE').Model.values[0] for h in HS}
OUT['best'] = best
DM = []
for h in HS:
    va, te = idx(h); o = y[te + h]
    for ref in ['Persistence', 'Climatology', 'STL-TBATS', 'TBATS', 'STL-ARIMA', 'SVR', 'LSTM', 'STL-TBATS-LSTM', 'STL-TBATS-SVR']:
        for cand in ['STL-TBATS-ENS']:
            if ref == cand: continue
            s, p = dm(o - P[cand][h]['te'], o - P[ref][h]['te'], h)
            DM.append(dict(h=h, model=cand, ref=ref, DM=s, p=p))
DM = pd.DataFrame(DM); DM.to_csv('dm_tests.csv', index=False); print(DM)

# ---- split-conformal 90% prediction intervals (calibrated on 2011-2013, by season) ----
def season(ix):
    mo = dates[ix].month.values
    return np.where(np.isin(mo, [3, 4, 5]), 'Pre-monsoon', np.where(np.isin(mo, [6, 7, 8, 9]), 'Monsoon',
             np.where(np.isin(mo, [10, 11]), 'Post-monsoon', 'Winter')))
CI = []; PI = {}
for m in ['Persistence', 'STL-TBATS', 'STL-TBATS-LSTM', 'STL-TBATS-SVR', 'STL-TBATS-ENS']:
    for h in HS:
        va, te = idx(h); rv = np.abs(y[va + h] - P[m][h]['va']); sv = season(va + h); st = season(te + h)
        q = {s: np.quantile(rv[sv == s], 0.9 * (1 + 1 / (sv == s).sum())) for s in np.unique(sv)}
        w = np.array([q[s] for s in st]); f = P[m][h]['te']; o = y[te + h]
        PI[(m, h)] = (f - w, f + w)
        CI.append(dict(Model=m, h=h, coverage=np.mean((o >= f - w) & (o <= f + w)) * 100, width=np.mean(2 * w)))
CI = pd.DataFrame(CI); CI.to_csv('conformal.csv', index=False); print(CI)

# ---- heat-warning skill (deterministic and probabilistic) ----
def cont(obs, fc):
    a = np.sum(obs & fc); b = np.sum(~obs & fc); c = np.sum(obs & ~fc); d_ = np.sum(~obs & ~fc); N = a + b + c + d_
    pod = a / (a + c) if a + c else np.nan; far = b / (a + b) if a + b else np.nan
    csi = a / (a + b + c) if a + b + c else np.nan
    hss = 2 * (a * d_ - b * c) / ((a + c) * (c + d_) + (a + b) * (b + d_))
    return dict(hits=a, false=b, miss=c, POD=pod, FAR=far, CSI=csi, HSS=hss)
WS = []
for thr, lab in [(41, 'Danger (HI>=41)'), (32, 'Extreme caution+ (HI>=32)')]:
    for m in ['Persistence', 'Climatology', 'STL-TBATS', 'SVR', 'LSTM', 'STL-TBATS-LSTM', 'STL-TBATS-SVR', 'STL-TBATS-ENS']:
        for h in HS:
            va, te = idx(h); o = y[te + h] >= thr; f = P[m][h]['te'] >= thr
            r = cont(o, f); r.update(Model=m, h=h, event=lab, base_rate=o.mean() * 100); WS.append(r)
    # probabilistic: conformal-based exceedance probability for ensemble -> warn if P>=0.3
    for m in ['STL-TBATS-ENS']:
        for h in HS:
            va, te = idx(h); res_va = y[va + h] - P[m][h]['va']
            f = P[m][h]['te']; prob = (f[:, None] + res_va[None, :] >= thr).mean(1)
            o = y[te + h] >= thr
            bs = np.mean((prob - o) ** 2); bs_clim = np.mean((o.mean() - o) ** 2)
            for pt in [0.3]:
                r = cont(o, prob >= pt); r.update(Model=f'{m} (P>={pt})', h=h, event=lab, base_rate=o.mean() * 100,
                                                  BSS=1 - bs / bs_clim); WS.append(r)
WS = pd.DataFrame(WS); WS.to_csv('warning_skill.csv', index=False)
print(WS[WS.event.str.startswith('Danger')].round(3).to_string())

# ---- seasonal breakdown of MAE for selected models (h=1,3) ----
SB = []
for m in ['Persistence', 'STL-TBATS', 'STL-TBATS-LSTM', 'STL-TBATS-ENS']:
    for h in HS:
        va, te = idx(h); se = season(te + h); e = np.abs(y[te + h] - P[m][h]['te'])
        for s in ['Winter', 'Pre-monsoon', 'Monsoon', 'Post-monsoon']:
            SB.append(dict(Model=m, h=h, Season=s, MAE=e[se == s].mean()))
SB = pd.DataFrame(SB); SB.to_csv('season_mae.csv', index=False)
print(SB.pivot_table(index=['Model', 'h'], columns='Season', values='MAE').round(3))

# ---- climate trends (1981-2024) ----
df = pd.DataFrame({'HI': y}, index=dates)
ann = pd.DataFrame({'mean': df.HI.resample('YE').mean(),
                    'MAM': df[df.index.month.isin([3, 4, 5])].HI.resample('YE').mean(),
                    'JJAS': df[df.index.month.isin([6, 7, 8, 9])].HI.resample('YE').mean(),
                    'p95': df.HI.resample('YE').quantile(0.95),
                    'd41': (df.HI >= 41).resample('YE').sum(), 'd32': (df.HI >= 32).resample('YE').sum()})
TR = []
for c in ann:
    r = mk.hamed_rao_modification_test(ann[c].values)
    TR.append(dict(var=c, slope_per_decade=r.slope * 10, p=r.p, tau=r.Tau, first10=ann[c].iloc[:10].mean(), last10=ann[c].iloc[-10:].mean()))
TR = pd.DataFrame(TR); TR.to_csv('trends.csv', index=False); print(TR)
ann.to_csv('annual.csv')

# ---- descriptive statistics & categories (raw, uncleaned) ----
cats = pd.cut(df.HI, [-np.inf, 27, 32, 41, 54, np.inf], right=False,
              labels=['Normal', 'Caution', 'Extreme caution', 'Danger', 'Extreme danger'])
OUT['desc'] = dict(mean=df.HI.mean(), sd=df.HI.std(), min=df.HI.min(), max=df.HI.max(),
                   dmin=str(df.HI.idxmin().date()), dmax=str(df.HI.idxmax().date()),
                   cats=(cats.value_counts(normalize=True) * 100).round(2).to_dict(),
                   test_d41=float((y[ntr:] >= 41).mean() * 100), train_d41=float((y[:ntr] >= 41).mean() * 100))

# ---- trend-adjusted climatological outlook for 2025-2026 (analogue years 2010-2024) ----
sl = TR.set_index('var').loc['mean', 'slope_per_decade'] / 10
OL = []
for target in [2025, 2026]:
    for Y in range(2010, 2025):
        v = df[df.index.year == Y].HI.values[:365] + sl * (target - Y)
        c = pd.cut(v, [-np.inf, 27, 32, 41, 54, np.inf], right=False,
                   labels=['Normal', 'Caution', 'Extreme caution', 'Danger', 'Extreme danger'])
        r = (pd.Series(c).value_counts(normalize=True) * 100).to_dict(); r.update(target=target, analogue=Y)
        mo = pd.date_range(f'{Y}-01-01', periods=365).month
        for mth in range(1, 13): r[f'd41_m{mth}'] = np.sum((v >= 41) & (mo == mth))
        OL.append(r)
OL = pd.DataFrame(OL); OL.to_csv('outlook.csv', index=False)
print(OL.groupby('target')[['Normal', 'Caution', 'Extreme caution', 'Danger']].describe(percentiles=[.05, .5, .95]).T.round(2))
_orig = ROOT / 'data' / 'original_projection_2025_2026.csv'   # optional: projection from the first submission
if _orig.exists():
    orig = pd.read_csv(_orig)
    OUT['orig_outlook'] = dict(EC=float(((orig.HeatIndex >= 32) & (orig.HeatIndex < 41)).mean() * 100), D=float((orig.HeatIndex >= 41).mean() * 100))
OUT['scale'] = scale; OUT['slope'] = sl
pickle.dump(dict(P=P, PI=PI, M=M), open('stage3.pkl', 'wb'))
json.dump(OUT, open('out.json', 'w'), indent=1, default=str)
print(json.dumps(OUT, indent=1, default=str))
