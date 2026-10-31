"""Stage 1: data, leak-free STL seasonal profile, linear (statistical) h-step rolling forecasts.
All parameters are estimated on the training period (1981-2013) only; the test period (2014-2024)
is processed with fixed parameters using information available at each forecast origin."""
from pathlib import Path as _P
import os as _os
ROOT = _P(__file__).resolve().parents[1]
RES = ROOT / 'results'; FIGDIR = ROOT / 'figures'
RES.mkdir(exist_ok=True); FIGDIR.mkdir(exist_ok=True)
_os.chdir(RES)  # intermediate files and tables are written to results/

import numpy as np, pandas as pd, itertools, warnings, pickle
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
warnings.filterwarnings("ignore")

HS = [1, 3, 7]
d = pd.read_excel(_os.environ.get('HI_DATA', str(ROOT / 'data' / 'Heat_Index_Data_For_Jessore.xlsx')))
d = d.set_index('Date').asfreq('D')
for c in ['T', 'RH', 'HeatIndex']:
    d[c] = d[c].interpolate()          # one missing day (1982-06-02)
y = d['HeatIndex'].values.astype(float)
n = len(y)
ntr = round(n * 0.75)                   # 12053 -> 1981-01-01..2013-12-31
nval_start = d.index.get_loc(pd.Timestamp('2011-01-01'))
dates = d.index
doy = np.minimum(dates.dayofyear.values, 365)
print('train end', dates[ntr - 1], 'test', dates[ntr], '->', dates[-1], 'n_test', n - ntr)

# ---- leak-free seasonal profile: STL on training data only, averaged by day of year ----
stl = STL(pd.Series(y[:ntr], index=dates[:ntr]), period=365, seasonal=7 * 52 + 1, robust=True).fit()
seas_tr = stl.seasonal.values
prof = pd.Series(seas_tr).groupby(doy[:ntr]).mean()
prof = prof.rolling(15, center=True, min_periods=1).mean()  # light smoothing
S = prof.reindex(doy).values
SA = y - S                                # seasonally adjusted series (train-based profile)

# climatology anomalies of T and RH (train-based DOY means)
def anom(v):
    m = pd.Series(v[:ntr]).groupby(doy[:ntr]).mean().rolling(15, center=True, min_periods=1).mean()
    return v - m.reindex(doy).values
Ta, RHa = anom(d['T'].values), anom(d['RH'].values.astype(float))

# ---------- ARMA h-step recursion with fixed parameters ----------
def arma_hstep(w, eps, phi, theta, h):
    """w: zero-mean series (len n), eps: one-step innovations. returns F[t] = forecast of w[t+h] made at t"""
    p, q = len(phi), len(theta)
    n = len(w); F = np.full(n, np.nan)
    # forecasts path matrix: for each origin t keep predictions k=1..h
    preds = np.zeros((n, h + 1))
    for k in range(1, h + 1):
        val = np.zeros(n)
        for i in range(1, p + 1):
            if k - i >= 1:
                val += phi[i - 1] * preds[:, k - i]
            else:
                val += phi[i - 1] * np.roll(w, -(k - i))   # w[t+k-i]
        for j in range(k, q + 1):
            val += theta[j - 1] * np.roll(eps, -(k - j))
        preds[:, k] = val
    F = preds[:, h]
    F[:max(p, q)] = np.nan
    return F  # aligned to origin t

def fit_arima_auto(z, max_p=3, max_q=2, ds=(0, 1)):
    best = None
    for dd, p, q in itertools.product(ds, range(max_p + 1), range(max_q + 1)):
        if p + q == 0 and dd == 0: continue
        try:
            r = ARIMA(z[:ntr], order=(p, dd, q), trend='c' if dd == 0 else 'n').fit()
            if best is None or r.aic < best[0]: best = (r.aic, (p, dd, q), r)
        except Exception: pass
    return best

def arima_forecasts(z, best):
    aic, (p, dd, q), r = best
    rf = r.apply(z)                     # fixed params, full series
    par = dict(zip(r.model.param_names, r.params))
    phi = np.array([par[f'ar.L{i}'] for i in range(1, p + 1)])
    theta = np.array([par[f'ma.L{i}'] for i in range(1, q + 1)])
    out = {}
    if dd == 0:
        mu = par['const']; w = z - mu; eps = rf.resid
        for h in HS: out[h] = arma_hstep(w, eps, phi, theta, h) + mu
    else:
        dz = np.r_[0, np.diff(z)]; eps = rf.resid
        for h in HS:
            cum = np.zeros(n)
            for k in range(1, h + 1):
                cum += arma_hstep(dz, eps, phi, theta, k)
            out[h] = z + cum
    return out, (p, dd, q), rf.resid

def ets_forecasts(z):
    best = None
    for tr, dp in [(None, False), ('add', True)]:
        m = ExponentialSmoothing(z[:ntr], trend=tr, damped_trend=dp, initialization_method='estimated').fit()
        k = 2 + (3 if tr else 0)
        aic = ntr * np.log(m.sse / ntr) + 2 * k
        if best is None or aic < best[0]: best = (aic, tr, m)
    aic, tr, m = best
    a = m.params['smoothing_level']; l = m.params['initial_level']
    b = m.params.get('initial_trend', 0.0) if tr else 0.0
    be = m.params.get('smoothing_trend', 0.0) if tr else 0.0
    ph = m.params.get('damping_trend', 1.0) if tr else 0.0
    L = np.zeros(n); B = np.zeros(n)
    for t in range(n):
        e = z[t] - (l + ph * b)
        l, b = l + ph * b + a * e, ph * b + be * e
        L[t], B[t] = l, b
    out = {}
    for h in HS:
        s = sum(ph ** i for i in range(1, h + 1)) if tr else 0
        out[h] = L + s * B
    return out, ('A,Ad,N' if tr else 'A,N,N', round(a, 3))

def fourier(t, K, P=365.25):
    return np.column_stack([f(2 * np.pi * k * t / P) for k in range(1, K + 1) for f in (np.sin, np.cos)])

# ---------- run linear models ----------
LIN = {}; INFO = {}
t = np.arange(n)
# Persistence and climatology baselines
for h in HS:
    LIN.setdefault('Persistence', {})[h] = y.copy()
    LIN.setdefault('Climatology', {})[h] = np.roll(S, -h) + SA[:ntr].mean()

# ARIMA on raw HI (non-seasonal, as in original)
b = fit_arima_auto(y); f, o, _ = arima_forecasts(y, b); LIN['ARIMA'] = f; INFO['ARIMA'] = o
# ETS on raw HI
f, o = ets_forecasts(y); LIN['ETS'] = f; INFO['ETS'] = o
# TBATS-type trigonometric seasonal model: Fourier(K) + linear trend + ARMA errors
bestK = None
for K in range(1, 7):
    X = np.column_stack([np.ones(n), t / 365.25, fourier(t, K)])
    beta, *_ = np.linalg.lstsq(X[:ntr], y[:ntr], rcond=None)
    rss = ((y[:ntr] - X[:ntr] @ beta) ** 2).sum(); aic = ntr * np.log(rss / ntr) + 2 * X.shape[1]
    if bestK is None or aic < bestK[0]: bestK = (aic, K, beta, X)
_, K, beta, X = bestK
det = X @ beta; u = y - det
b = fit_arima_auto(u, ds=(0,)); fu, o, _ = arima_forecasts(u, b)
LIN['TBATS'] = {h: np.roll(det, -h) + fu[h] for h in HS}; INFO['TBATS'] = ('K=%d' % K, o)
# STL-TBATS: trigonometric (Fourier) + trend + ARMA model fitted to the STL seasonally adjusted series
bestK2 = None
for K2 in range(0, 4):
    X2 = np.column_stack([np.ones(n), t / 365.25] + ([fourier(t, K2)] if K2 else []))
    beta2, *_ = np.linalg.lstsq(X2[:ntr], SA[:ntr], rcond=None)
    rss = ((SA[:ntr] - X2[:ntr] @ beta2) ** 2).sum(); aic = ntr * np.log(rss / ntr) + 2 * X2.shape[1]
    if bestK2 is None or aic < bestK2[0]: bestK2 = (aic, K2, beta2, X2)
_, K2, beta2, X2 = bestK2
det2 = X2 @ beta2; u2 = SA - det2
b = fit_arima_auto(u2, ds=(0,)); fu2, o, _ = arima_forecasts(u2, b)
LIN['STL-TBATS'] = {h: np.roll(det2 + S, -h) + fu2[h] for h in HS}; INFO['STL-TBATS'] = ('K=%d' % K2, o)
# STL-based linear models on seasonally adjusted series
b = fit_arima_auto(SA); f, o, _ = arima_forecasts(SA, b)
LIN['STL-ARIMA'] = {h: f[h] + np.roll(S, -h) for h in HS}; INFO['STL-ARIMA'] = o
f, o = ets_forecasts(SA)
LIN['STL-ETS'] = {h: f[h] + np.roll(S, -h) for h in HS}; INFO['STL-ETS'] = o
print(INFO)

def target(h):
    v = np.full(n, np.nan); v[:n - h] = y[h:]; return v

for m in LIN:
    for h in HS:
        e = target(h)[ntr:] - LIN[m][h][ntr:]
        ok = ~np.isnan(e)
        print(f'{m:12s} h={h} MAE={np.abs(e[ok]).mean():.3f}')

pickle.dump(dict(d=d, y=y, n=n, ntr=ntr, nval_start=nval_start, dates=dates, doy=doy, S=S, SA=SA,
                 Ta=Ta, RHa=RHa, LIN=LIN, INFO=INFO, HS=HS, stl_trend=stl.trend.values, prof=prof),
            open('stage1.pkl', 'wb'))
