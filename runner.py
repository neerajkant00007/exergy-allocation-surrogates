"""Resumable experiment runner.  Usage: python3 runner.py <stage> [args]

stages:  setup | main <i0> <i1> | curve <n> | finish
Every chunk writes its own checkpoint file, so the run can be resumed.
"""
import warnings, json, os, sys, time, pickle, numpy as np
warnings.filterwarnings('ignore')
import model as M
import surrogates as S
from scipy import stats

EPS = 1e-2
N_MC = 10000
N_TRAIN, N_TEST, N_COLL = 240, 240, 200
R_MAIN, R_CURVE = 8, 3
CURVE_SIZES = [60, 120, 240, 480]
NN_EPOCHS, NN_EPOCHS_CURVE = 10000, 5000
VARIANTS = ['GP-PI', 'GP-plain', 'NN-hard', 'NN-soft', 'NN-plain']
CK = 'ck'
os.makedirs(CK, exist_ok=True)


def bin_of(X):
    return np.array([M.reference_model(x)['B_in'] for x in X])


def metrics(pred, Yref, Bin):
    lam = S.allocation(pred); lref = S.allocation(Yref)
    err = lam - lref
    r = S.balance_residual(pred, Bin); D = S.destruction(pred, Bin)
    ss = ((lref[:, 0] - lref[:, 0].mean()) ** 2).sum()
    return dict(mae=100 * float(np.abs(err).mean()), maxerr=100 * float(np.abs(err).max()),
                r2_el=float(1 - ((err[:, 0]) ** 2).sum() / ss),
                rmed=float(np.median(np.abs(r))),
                rej=100 * float(np.mean((np.abs(r) > EPS) | (D < 0))),
                negD=int((D < 0).sum()))


def load(name):
    with open(os.path.join(CK, name), 'rb') as fh:
        return pickle.load(fh)


def save(name, obj):
    with open(os.path.join(CK, name), 'wb') as fh:
        pickle.dump(obj, fh)


def build(Xtr, Ytr, Btr, Xc, Bc, seed, epochs, HYP, f):
    out = {}
    for name, phys in [('GP-PI', True), ('GP-plain', False)]:
        t = time.perf_counter()
        gp = S.GPSurrogate(HYP, physics=phys).fit(f(Xtr), Ytr, f(Xc), Bc)
        out[name] = (lambda X, B, g=gp: g.predict(f(X)), time.perf_counter() - t, gp)
    for name, phys, pen in [('NN-hard', True, True), ('NN-soft', False, True),
                            ('NN-plain', False, False)]:
        t = time.perf_counter()
        nn = S.NNSurrogate(Xtr.shape[1], physics=phys, penalty=pen, seed=seed,
                           epochs=epochs, lr=3e-3, mu0=1.0, mu_max=1e2,
                           outer_every=1000).fit(f(Xtr), Ytr, Btr, f(Xc), Bc)
        out[name] = (lambda X, B, m=nn: m.predict(f(X), B), time.perf_counter() - t, nn)
    return out


# ------------------------------------------------------------------- setup
if sys.argv[1] == 'setup':
    RNG = np.random.default_rng(2026)
    X_mc = M.sample_inputs(N_MC, RNG)
    t0 = time.perf_counter(); Y_mc = M.evaluate(X_mc); t_mc = time.perf_counter() - t0
    B_mc = bin_of(X_mc); lam_mc = S.allocation(Y_mc)
    X_h = M.sample_inputs(480, RNG, lhs=True); Y_h = M.evaluate(X_h)
    mu_x, sd_x = X_h.mean(0), X_h.std(0) + 1e-12
    t0 = time.perf_counter()
    HYP = S.fit_gp_hypers((X_h - mu_x) / sd_x, Y_h)
    t_hyp = time.perf_counter() - t0
    bs = []
    for _ in range(2000):
        idx = RNG.integers(0, N_MC, N_MC); c = np.corrcoef(lam_mc[idx].T)
        bs.append([c[0, 1], c[0, 2], c[1, 2]])
    bs = np.array(bs)
    nom = M.reference_model(M.nominal_input())
    save('setup.pkl', dict(X_mc=X_mc, Y_mc=Y_mc, B_mc=B_mc, lam_mc=lam_mc, HYP=HYP,
                           mu_x=mu_x, sd_x=sd_x, t_mc=t_mc, t_hyp=t_hyp,
                           corr=np.corrcoef(lam_mc.T), corr_ci=np.percentile(bs, [2.5, 97.5], 0),
                           nominal={k: float(v) for k, v in nom.items()}))
    print('setup done: MC %.3f s (%.4f ms/sample), hypers %.0f s' %
          (t_mc, 1e3 * t_mc / N_MC, t_hyp))
    print('lam mean', lam_mc.mean(0).round(4), 'sd', lam_mc.std(0).round(4))

# -------------------------------------------------------------- main repetitions
elif sys.argv[1] == 'main':
    s = load('setup.pkl'); HYP = s['HYP']
    f = lambda A: (A - s['mu_x']) / s['sd_x']
    i0, i1 = int(sys.argv[2]), int(sys.argv[3])
    for r_i in range(i0, i1):
        rng = np.random.default_rng(1000 + r_i)
        Xtr = M.sample_inputs(N_TRAIN, rng, lhs=True); Ytr = M.evaluate(Xtr); Btr = bin_of(Xtr)
        Xc = M.sample_inputs(N_COLL, rng); Bc = bin_of(Xc)
        Xte = M.sample_inputs(N_TEST, rng); Yte = M.evaluate(Xte); Bte = bin_of(Xte)
        V = build(Xtr, Ytr, Btr, Xc, Bc, r_i, NN_EPOCHS, HYP, f)
        rec = {}
        for v in VARIANTS:
            pf, tfit, obj = V[v]
            pred = pf(Xte, Bte)
            m = metrics(pred, Yte, Bte); m['tfit'] = tfit
            pred_mc = pf(s['X_mc'], s['B_mc']); lam_s = S.allocation(pred_mc)
            ks = [stats.ks_2samp(lam_s[:, j], s['lam_mc'][:, j]) for j in range(3)]
            m['ks'] = [[float(k.statistic), float(k.pvalue)] for k in ks]
            rec[v] = m
            if r_i == 0:
                rec.setdefault('_arrays', {})[v] = dict(
                    pred_te=pred, lam_mc=lam_s,
                    resid=S.balance_residual(pred_mc, s['B_mc']),
                    destr=S.destruction(pred_mc, s['B_mc']))
                if v == 'GP-PI':
                    rec['_al'] = obj.history
                if v == 'NN-hard':
                    rec['_nn'] = obj.history
        if r_i == 0:
            rec['_te'] = dict(Yte=Yte, Bte=Bte)
        save('main_%d.pkl' % r_i, rec)
        print('rep %d done  GP-PI %.4f  NN-soft %.4f' %
              (r_i, rec['GP-PI']['mae'], rec['NN-soft']['mae']), flush=True)

# --------------------------------------------------------------- learning curve
elif sys.argv[1] == 'curve':
    s = load('setup.pkl'); HYP = s['HYP']
    f = lambda A: (A - s['mu_x']) / s['sd_x']
    n = int(sys.argv[2]); out = {v: [] for v in VARIANTS}
    for r_i in range(R_CURVE):
        rng = np.random.default_rng(5000 + 100 * n + r_i)
        Xtr = M.sample_inputs(n, rng, lhs=True); Ytr = M.evaluate(Xtr); Btr = bin_of(Xtr)
        Xc = M.sample_inputs(N_COLL, rng); Bc = bin_of(Xc)
        Xte = M.sample_inputs(N_TEST, rng); Yte = M.evaluate(Xte); Bte = bin_of(Xte)
        V = build(Xtr, Ytr, Btr, Xc, Bc, r_i, NN_EPOCHS_CURVE, HYP, f)
        for v in VARIANTS:
            out[v].append(metrics(V[v][0](Xte, Bte), Yte, Bte)['mae'])
        print('  n=%d rep %d' % (n, r_i), flush=True)
    save('curve_%d.pkl' % n, out)
    print('curve n=%d done' % n)

# --------------------------------------------------------------------- finish
elif sys.argv[1] == 'finish':
    s = load('setup.pkl')
    res = dict(mc=dict(n=N_MC, t_total_s=s['t_mc'], t_per_sample_ms=1e3 * s['t_mc'] / N_MC,
                       lam_mean=s['lam_mc'].mean(0).tolist(),
                       lam_sd=s['lam_mc'].std(0).tolist(),
                       corr=s['corr'].tolist(), corr_ci=s['corr_ci'].tolist()),
               nominal=s['nominal'], gp_hyper_fit_s=s['t_hyp'],
               settings=dict(eps=EPS, n_train=N_TRAIN, n_test=N_TEST, n_coll=N_COLL,
                             r_main=R_MAIN, r_curve=R_CURVE, nn_epochs=NN_EPOCHS,
                             nn_epochs_curve=NN_EPOCHS_CURVE))
    reps = [load('main_%d.pkl' % i) for i in range(R_MAIN)]
    res['main'] = {}
    for v in VARIANTS:
        d = {}
        for k in ['mae', 'maxerr', 'r2_el', 'rmed', 'rej', 'tfit']:
            arr = np.array([r[v][k] for r in reps], float)
            hw = 1.96 * arr.std(ddof=1) / np.sqrt(len(arr))
            d[k] = dict(mean=float(arr.mean()), ci=[float(arr.mean() - hw), float(arr.mean() + hw)],
                        sd=float(arr.std(ddof=1)))
        ks = np.array([r[v]['ks'] for r in reps])
        d['ks_D'] = ks[:, :, 0].mean(0).tolist()
        d['ks_p_median'] = np.median(ks[:, :, 1], 0).tolist()
        d['ks_reject_frac'] = (ks[:, :, 1] < 0.05).mean(0).tolist()
        d['negD_total'] = int(sum(r[v]['negD'] for r in reps))
        res['main'][v] = d
    res['curve'] = {}
    for n in CURVE_SIZES:
        c = load('curve_%d.pkl' % n)
        for v in VARIANTS:
            a = np.array(c[v], float)
            res['curve'].setdefault(v, {})[str(n)] = dict(
                mean=float(a.mean()), ci=float(1.96 * a.std(ddof=1) / np.sqrt(len(a))))
    A = reps[0]['_arrays']
    np.savez_compressed('arrays_part1.npz', lam_mc=s['lam_mc'], B_mc=s['B_mc'],
                        Yte=reps[0]['_te']['Yte'], Bte=reps[0]['_te']['Bte'],
                        **{('lam_' + v): A[v]['lam_mc'] for v in VARIANTS},
                        **{('resid_' + v): A[v]['resid'] for v in VARIANTS},
                        **{('destr_' + v): A[v]['destr'] for v in VARIANTS},
                        **{('predte_' + v): A[v]['pred_te'] for v in VARIANTS})
    json.dump(dict(al=reps[0]['_al'], nn=reps[0]['_nn']), open('hist.json', 'w'))
    json.dump(res, open('results_part1.json', 'w'), indent=1)
    for v in VARIANTS:
        d = res['main'][v]
        print('%-9s MAE %.4f [%.4f, %.4f] pp  rej %.2f%%  KS D %s' %
              (v, d['mae']['mean'], d['mae']['ci'][0], d['mae']['ci'][1],
               d['rej']['mean'], np.round(d['ks_D'], 4)))
