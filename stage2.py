"""Stage 2: ML models (single, STL-adjusted, and residual hybrids) for direct h-step forecasts.
Train origins: 1981-2010; validation origins (early stopping / calibration): 2011-2013; test: 2014-2024."""
from pathlib import Path as _P
import os as _os
ROOT = _P(__file__).resolve().parents[1]
RES = ROOT / 'results'; FIGDIR = ROOT / 'figures'
RES.mkdir(exist_ok=True); FIGDIR.mkdir(exist_ok=True)
_os.chdir(RES)  # intermediate files and tables are written to results/

import numpy as np, pandas as pd, pickle, time, sys, warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb, torch, torch.nn as nn
warnings.filterwarnings("ignore")
torch.set_num_threads(2)
D = pickle.load(open('stage1.pkl', 'rb'))
y, n, ntr, nv, S, SA, Ta, RHa, LIN, HS, doy = [D[k] for k in
    ['y', 'n', 'ntr', 'nval_start', 'S', 'SA', 'Ta', 'RHa', 'LIN', 'HS', 'doy']]

def lag(v, k): return np.r_[np.full(k, np.nan), v[:-k]] if k else v.copy()
def lead(v, h): return np.r_[v[h:], np.full(h, np.nan)]

CONFIGS = {  # name prefix -> stage-1 linear model (None = no linear stage)
    '': None, 'STL': 'STL', 'ARIMA': 'ARIMA', 'TBATS': 'TBATS',
    'STL-ARIMA': 'STL-ARIMA', 'STL-ETS': 'STL-ETS', 'STL-TBATS': 'STL-TBATS'}

def build(cfg, h):
    if cfg is None:
        x = y; base = np.zeros(n); tgt = lead(y, h)
    elif cfg == 'STL':
        x = SA; base = lead(S, h); tgt = lead(SA, h)
    else:
        F = LIN[cfg]
        x = y - lag(F[1], 1)              # one-step residuals known at origin
        base = F[h]; tgt = lead(y, h) - F[h]
    feats = {}
    for k in range(7): feats[f'x_lag{k}'] = lag(x, k)
    feats['x_mean7'] = pd.Series(x).rolling(7).mean().values
    feats['x_mean30'] = pd.Series(x).rolling(30).mean().values
    if cfg is not None:
        for k in range(4): feats[f'HIanom_lag{k}'] = lag(SA, k)
    for k in range(3):
        feats[f'Tanom_lag{k}'] = lag(Ta, k); feats[f'RHanom_lag{k}'] = lag(RHa, k)
    dt = lead(doy.astype(float), h); dt[np.isnan(dt)] = 1
    feats['sin_doy'] = np.sin(2 * np.pi * dt / 365.25); feats['cos_doy'] = np.cos(2 * np.pi * dt / 365.25)
    if cfg not in (None, 'STL'):
        feats['linear_fc_anom'] = F[h] - lead(S, h)
    X = pd.DataFrame(feats)
    seq = np.column_stack([x, SA, Ta, RHa])     # for LSTM sequences
    return X, tgt, base, seq

def idx(h):
    tr = np.arange(40, nv - h); va = np.arange(nv, ntr - h); te = np.arange(ntr, n - h)
    return tr, va, te

class LSTMNet(nn.Module):
    def __init__(s, nin, nstat, hid=32):
        super().__init__(); s.l = nn.LSTM(nin, hid, batch_first=True)
        s.o = nn.Sequential(nn.Linear(hid + nstat, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(s, a, b):
        _, (hh, _) = s.l(a); return s.o(torch.cat([hh[-1], b], 1)).squeeze(1)

def fit_lstm(seq, X, tgt, tr, va, te, L=30, seed=1):
    torch.manual_seed(seed); np.random.seed(seed)
    sc = StandardScaler().fit(seq[tr]); sq = np.nan_to_num(sc.transform(np.nan_to_num(seq)))
    stat_cols = [c for c in X.columns if c in ('sin_doy', 'cos_doy', 'linear_fc_anom')]
    st = X[stat_cols].values; ss = StandardScaler().fit(st[tr]); st = ss.transform(st)
    ym, ysd = tgt[tr].mean(), tgt[tr].std()
    def mk(ix):
        a = np.stack([sq[i - L + 1:i + 1] for i in ix]).astype(np.float32)
        return torch.tensor(a), torch.tensor(st[ix].astype(np.float32))
    A_tr, B_tr = mk(tr); Y_tr = torch.tensor(((tgt[tr] - ym) / ysd).astype(np.float32))
    A_va, B_va = mk(va); Y_va = torch.tensor(((tgt[va] - ym) / ysd).astype(np.float32))
    net = LSTMNet(seq.shape[1], len(stat_cols)); opt = torch.optim.Adam(net.parameters(), 1e-3)
    best, bstate, wait = 1e9, None, 0
    for ep in range(60):
        net.train(); perm = torch.randperm(len(tr))
        for i in range(0, len(tr), 256):
            j = perm[i:i + 256]; opt.zero_grad()
            loss = ((net(A_tr[j], B_tr[j]) - Y_tr[j]) ** 2).mean(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): v = ((net(A_va, B_va) - Y_va) ** 2).mean().item()
        if v < best - 1e-4: best, bstate, wait = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= 6: break
    net.load_state_dict(bstate); net.eval()
    out = {}
    for nm, ix in [('va', va), ('te', te)]:
        A, B = mk(ix)
        with torch.no_grad(): out[nm] = net(A, B).numpy() * ysd + ym
    return out, ep + 1

def fit_all(X, tgt, seq, h, which):
    tr, va, te = idx(h); P = {}
    Xtr, Xva, Xte = X.values[tr], X.values[va], X.values[te]
    ok = ~np.isnan(Xtr).any(1); Xtr_, ytr_ = Xtr[ok], tgt[tr][ok]
    if 'RF' in which:
        m = RandomForestRegressor(n_estimators=300, min_samples_leaf=10, max_features=0.33, n_jobs=2, random_state=1).fit(Xtr_, ytr_)
        P['RF'] = dict(va=m.predict(Xva), te=m.predict(Xte), model=m)
    if 'LGBM' in which:
        m = lgb.LGBMRegressor(n_estimators=3000, learning_rate=0.02, num_leaves=15, min_child_samples=40,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, random_state=1)
        m.fit(Xtr_, ytr_, eval_set=[(Xva, tgt[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
        P['LGBM'] = dict(va=m.predict(Xva), te=m.predict(Xte), model=m)
    sc = StandardScaler().fit(Xtr_)
    if 'SVR' in which:
        m = SVR(C=3.0, epsilon=0.1, gamma='scale').fit(sc.transform(Xtr_), ytr_)
        P['SVR'] = dict(va=m.predict(sc.transform(Xva)), te=m.predict(sc.transform(Xte)))
    if 'ANN' in which:
        m = MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1e-3, learning_rate_init=1e-3, max_iter=300, early_stopping=True,
                         validation_fraction=0.1, n_iter_no_change=15, random_state=1).fit(sc.transform(Xtr_), ytr_)
        P['ANN'] = dict(va=m.predict(sc.transform(Xva)), te=m.predict(sc.transform(Xte)))
    if 'LSTM' in which:
        o, ep = fit_lstm(seq, X, tgt, tr, va, te); P['LSTM'] = dict(va=o['va'], te=o['te'], epochs=ep)
    return P

if __name__ == '__main__':
    which = ['RF', 'LGBM', 'SVR', 'ANN', 'LSTM']
    R = {}; MODELS = {}
    for pre, cfg in CONFIGS.items():
        for h in HS:
            t0 = time.time()
            X, tgt, base, seq = build(cfg, h)
            P = fit_all(X, tgt, seq, h, which)
            tr, va, te = idx(h)
            for ml, p in P.items():
                name = f'{pre}-{ml}' if pre else ml
                R.setdefault(name, {})[h] = dict(va=base[va] + p['va'], te=base[te] + p['te'])
                if ml in ('LGBM', 'RF'):
                    MODELS[(name, h)] = (p['model'], X.columns.tolist())
                e = (y[te + h] - R[name][h]['te']); print(f'{name:18s} h={h} MAE={np.abs(e).mean():.3f}', flush=True)
            print(f'  [{pre or "raw"} h={h} {time.time()-t0:.0f}s]', flush=True)
        pickle.dump(dict(R=R), open('stage2.pkl', 'wb'))
    pickle.dump(dict(R=R), open('stage2.pkl', 'wb'))
    pickle.dump(MODELS, open('stage2_models.pkl', 'wb'))
