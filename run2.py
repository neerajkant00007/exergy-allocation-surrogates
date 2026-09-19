"""Part 2: sensitivity analysis, adaptive sampling, validator fallback, timings."""
import warnings, json, time, numpy as np
warnings.filterwarnings('ignore')
import model as M
import surrogates as S
from scipy.stats import truncnorm, qmc

RNG = np.random.default_rng(7777)
EPS = 1e-2
res = json.load(open('results_part1.json'))
arr = dict(np.load('arrays_part1.npz'))

# ---------------------------------------------------------------- independent sampler
# Sobol' analysis assumes independent factors: the Dirichlet composition is
# generated from 12 independent Gamma variates (grouped as one factor), and the
# continuous inputs are sampled from their marginals without the copula.
GAM_A = M.DIRICHLET_S * M.W_NOM
GROUPS = [("Waste composition", list(range(12)))] + \
         [(n, [12 + k]) for k, n in enumerate(M.CONT_NAMES)]
NFAC = 12 + len(M.CONT)


def u_to_x(U):
    """map U in [0,1]^NFAC to model inputs (independent factors)."""
    from scipy.special import gammaincinv
    g = np.column_stack([gammaincinv(GAM_A[i], np.clip(U[:, i], 1e-9, 1 - 1e-9))
                         for i in range(12)])
    w = g / g.sum(1, keepdims=True)
    cols = []
    for k, (_, kind, par) in enumerate(M.CONT):
        u = np.clip(U[:, 12 + k], 1e-9, 1 - 1e-9)
        if kind == "uniform":
            a, b = par
            cols.append(a + (b - a) * u)
        else:
            mu, sd, lo, hi = par
            a, b = (lo - mu) / sd, (hi - mu) / sd
            cols.append(truncnorm.ppf(u, a, b, loc=mu, scale=sd))
    return np.hstack([w, np.column_stack(cols)])


def lam_of(X):
    Y = M.evaluate(X)
    return S.allocation(Y)


# --------------------------------------------------------- Saltelli Sobol' indices
NS = 4096
A = qmc.Sobol(d=2 * NFAC, scramble=True, seed=11).random(NS)
Au, Bu = A[:, :NFAC], A[:, NFAC:]
t0 = time.perf_counter()
YA, YB = lam_of(u_to_x(Au)), lam_of(u_to_x(Bu))
S1 = {}; ST = {}
for gname, idx in GROUPS:
    Cu = Au.copy(); Cu[:, idx] = Bu[:, idx]
    YC = lam_of(u_to_x(Cu))          # B-values in group, A elsewhere
    s1, st = [], []
    for j in range(3):
        va = YA[:, j].var()
        s1.append(float(np.mean(YB[:, j] * (YC[:, j] - YA[:, j])) / va))
        st.append(float(np.mean((YA[:, j] - YC[:, j]) ** 2) / (2 * va)))
    S1[gname] = s1; ST[gname] = st
res['sobol'] = dict(N=NS, S1=S1, ST=ST, t_s=time.perf_counter() - t0,
                    n_evals=int(NS * (len(GROUPS) + 2)))
print('Sobol done %.1f s' % res['sobol']['t_s'], flush=True)

# ------------------------------------------------------------- tornado (1-sigma)
xn = M.nominal_input()
base = S.allocation(M.evaluate(xn[None, :]))[0]
tor = {}
for gname, idx in GROUPS:
    lo, hi = xn.copy(), xn.copy()
    if gname == "Waste composition":
        # +/- 1 sd of the mixture LHV along the dominant Dirichlet direction
        sd = np.sqrt(M.W_NOM * (1 - M.W_NOM) / (M.DIRICHLET_S + 1))
        d = sd * np.sign(M.LHV - float(M.W_NOM @ M.LHV))
        hi[:12] = np.clip(M.W_NOM + d, 1e-6, None); hi[:12] /= hi[:12].sum()
        lo[:12] = np.clip(M.W_NOM - d, 1e-6, None); lo[:12] /= lo[:12].sum()
    else:
        k = idx[0]
        kind, par = M.CONT[k - 12][1], M.CONT[k - 12][2]
        if kind == "uniform":
            m, sd = 0.5 * (par[0] + par[1]), (par[1] - par[0]) / np.sqrt(12)
        else:
            m, sd = par[0], par[1]
        lo[k], hi[k] = m - sd, m + sd
    ylo = S.allocation(M.evaluate(lo[None, :]))[0]
    yhi = S.allocation(M.evaluate(hi[None, :]))[0]
    tor[gname] = dict(lo=(100 * (ylo - base)).tolist(), hi=(100 * (yhi - base)).tolist())
res['tornado'] = dict(base=base.tolist(), swings=tor)

# ------------------------------------- importance scores gamma_p (Eq. 9, finite diff)
gam = {}
for gname, idx in GROUPS:
    s = max(abs(tor[gname]['hi'][0]), abs(tor[gname]['lo'][0]))
    gam[gname] = float(s / 100.0)
res['gamma'] = gam

# --------------------------------------------------- adaptive (active) sampling
def gp_fit_predict(Xtr, Ytr, Xq):
    from surrogates import fit_gp_hypers, GPSurrogate
    hyp = HYP
    gp = GPSurrogate(hyp, physics=False).fit((Xtr - mu_x) / sd_x, Ytr)
    return gp


X_h = M.sample_inputs(480, RNG, lhs=True); Y_h = M.evaluate(X_h)
mu_x, sd_x = X_h.mean(0), X_h.std(0) + 1e-12
f = lambda Aa: (Aa - mu_x) / sd_x
HYP = S.fit_gp_hypers(f(X_h), Y_h)

def gp_std(gp, Xs, chunk=500):
    """GP posterior sd of the electricity-exergy output (output 0)."""
    h = gp.hyp[0]
    K = S.matern52(gp.X, gp.X, h['ls'], h['sf2']) + h['noise'] * np.eye(len(gp.X))
    L = np.linalg.cholesky(K)
    out = np.empty(len(Xs))
    for i in range(0, len(Xs), chunk):
        sl = slice(i, min(i + chunk, len(Xs)))
        Ks = S.matern52(Xs[sl], gp.X, h['ls'], h['sf2'])
        v = np.linalg.solve(L, Ks.T)
        out[sl] = np.sqrt(np.maximum(h['sf2'] - (v ** 2).sum(0), 0)) * h['sd']
    return out


Xte = M.sample_inputs(240, RNG); Yte = M.evaluate(Xte); Bte = np.array([M.reference_model(x)['B_in'] for x in Xte])
lref = S.allocation(Yte)
curve_act, curve_lhs, batch_hist, sd_hist = [], [], [], []
rng = np.random.default_rng(31)
N0, NMAX = 60, 240
Xtr = M.sample_inputs(N0, rng, lhs=True); Ytr = M.evaluate(Xtr)
pool = M.sample_inputs(3000, rng)
beta = 400.0
n_now = N0
while n_now < NMAX:
    gp = S.GPSurrogate(HYP, physics=False).fit(f(Xtr), Ytr)
    sd_pool = gp_std(gp, f(pool))
    nb = int(np.clip(np.ceil(beta * sd_pool.max()), 10, 40))
    nb = min(nb, NMAX - n_now)
    pick = np.argsort(-sd_pool)[:nb]
    Xtr = np.vstack([Xtr, pool[pick]]); Ytr = np.vstack([Ytr, M.evaluate(pool[pick])])
    pool = np.delete(pool, pick, axis=0)
    n_now += nb
    mae = 100 * np.abs(S.allocation(gp.predict(f(Xte))) - lref).mean()
    batch_hist.append(int(nb)); sd_hist.append(float(sd_pool.max()))
    curve_act.append([n_now, float(mae)])
    # matched LHS budget
    Xl = M.sample_inputs(n_now, rng, lhs=True); Yl = M.evaluate(Xl)
    gl = S.GPSurrogate(HYP, physics=False).fit(f(Xl), Yl)
    curve_lhs.append([n_now, float(100 * np.abs(S.allocation(gl.predict(f(Xte))) - lref).mean())])
res['active'] = dict(active=curve_act, lhs=curve_lhs, batches=batch_hist, sd=sd_hist,
                     N0=N0, beta=beta)
print('active learning done', flush=True)

# -------------------------------------------------- validator fallback on the MC set
lam_mc = arr['lam_mc']
fallback = {}
for v in ['GP-PI', 'GP-plain', 'NN-hard', 'NN-soft', 'NN-plain']:
    r = arr['resid_' + v]; D = arr['destr_' + v]; lam = arr['lam_' + v]
    rej = (np.abs(r) > EPS) | (D < 0)
    lam_after = lam.copy(); lam_after[rej] = lam_mc[rej]      # reference re-evaluation
    fallback[v] = dict(
        rej_rate=100 * float(rej.mean()),
        mae_before=100 * float(np.abs(lam - lam_mc).mean()),
        mae_after=100 * float(np.abs(lam_after - lam_mc).mean()),
        max_before=100 * float(np.abs(lam - lam_mc).max()),
        max_after=100 * float(np.abs(lam_after - lam_mc).max()),
        rej_vs_eps=[[float(e), 100 * float(((np.abs(r) > e) | (D < 0)).mean())]
                    for e in np.logspace(-4, -1, 16)],
        corr_after=np.corrcoef(lam_after.T).tolist())
res['fallback'] = fallback

# ------------------------------------------------------------------- timings
Xt = M.sample_inputs(2000, RNG)
t0 = time.perf_counter(); M.evaluate(Xt); t_ref = (time.perf_counter() - t0) / 2000
Xtr = M.sample_inputs(240, RNG, lhs=True); Ytr = M.evaluate(Xtr); Btr = np.array([M.reference_model(x)['B_in'] for x in Xtr])
Xc = M.sample_inputs(200, RNG); Bc = np.array([M.reference_model(x)['B_in'] for x in Xc])
gp = S.GPSurrogate(HYP, physics=True).fit(f(Xtr), Ytr, f(Xc), Bc)
Bt2 = np.array([M.reference_model(x)['B_in'] for x in Xt])
t0 = time.perf_counter(); gp.predict(f(Xt)); t_gp = (time.perf_counter() - t0) / 2000
nn = S.NNSurrogate(Xtr.shape[1], physics=True, seed=0, epochs=2000).fit(f(Xtr), Ytr, Btr, f(Xc), Bc)
t0 = time.perf_counter(); nn.predict(f(Xt), Bt2); t_nn = (time.perf_counter() - t0) / 2000
res['timing'] = dict(t_ref_ms=1e3 * t_ref, t_gp_ms=1e3 * t_gp, t_nn_ms=1e3 * t_nn,
                     n_train=240, n_coll=200, n_test=240, n_mc=10000)
json.dump(res, open('results.json', 'w'), indent=1)
np.savez_compressed('arrays_part2.npz', sd_hist=np.array(sd_hist), batch=np.array(batch_hist))
print('PART 2 COMPLETE', flush=True)
