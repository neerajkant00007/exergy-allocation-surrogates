"""Physics-informed and unconstrained surrogates for the WtE exergy model.

Outputs predicted: y = [B_el, B_heat, B_ash, B_L, S_gen]
Constraint (exergy balance, Eq. 2):  B_el + B_heat + B_ash + B_L + T0*S_gen/1e3 = B_in
written as  s . y = B_in  with s = [1, 1, 1, 1, T0/1e3].
"""
import numpy as np
from model import T0

S_VEC = np.array([1.0, 1.0, 1.0, 1.0, T0 / 1e3])
NOUT = 5


# --------------------------------------------------------------- GP kernels
def matern52(XA, XB, ls, sf2):
    d = (XA[:, None, :] - XB[None, :, :]) / ls
    r = np.sqrt(np.maximum((d ** 2).sum(-1), 1e-300))
    s5r = np.sqrt(5.0) * r
    return sf2 * (1.0 + s5r + (5.0 / 3.0) * r ** 2) * np.exp(-s5r)


def fit_gp_hypers(Xs, Y, seed=0):
    """Marginal-likelihood hyperparameters per output (sklearn), shared by both variants."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
    hyp = []
    for o in range(Y.shape[1]):
        y = Y[:, o]
        mu, sd = y.mean(), y.std()
        k = (ConstantKernel(1.0, (1e-3, 1e4))
             * Matern(length_scale=np.ones(Xs.shape[1]), nu=2.5,
                      length_scale_bounds=(1e-2, 1e4))
             + WhiteKernel(1e-4, (1e-6, 1e1)))
        gp = GaussianProcessRegressor(kernel=k, normalize_y=False, alpha=0.0,
                                      n_restarts_optimizer=0, random_state=seed)
        gp.fit(Xs, (y - mu) / sd)
        p = gp.kernel_.get_params()
        sf2 = p['k1__k1__constant_value']
        ls = np.atleast_1d(p['k1__k2__length_scale']).copy()
        noise = max(p['k2__noise_level'], 1e-6)   # jitter floor for a stable dual solve
        hyp.append(dict(ls=ls, sf2=sf2, noise=noise, mu=mu, sd=sd))
    return hyp


class GPSurrogate:
    """GP posterior mean, optionally trained with an augmented-Lagrangian
    penalty on the exergy-balance residual at collocation inputs."""

    def __init__(self, hyp, physics=True, rho=1.6, mu0=1.0, mu_max=1e4, n_outer=25,
                 tol=1e-6):
        self.hyp, self.physics = hyp, physics
        self.rho, self.mu0, self.mu_max, self.n_outer, self.tol = rho, mu0, mu_max, n_outer, tol
        self.history = []

    def fit(self, Xs, Y, Xc_s=None, Bin_c=None):
        """Penalised-likelihood GP mean.

        Unconstrained: standard GP posterior mean on the training points.
        Physics-informed: the representer expansion is extended to the
        collocation inputs, and the augmented Lagrangian of Eq. (7) is solved
        for the dual weights at each outer iteration (the residual is linear in
        the predictions, so each inner problem is a linear system).
        """
        n = len(Xs)
        self.hyp_used = self.hyp
        self.Yn = [(Y[:, o] - self.hyp[o]['mu']) / self.hyp[o]['sd'] for o in range(NOUT)]
        if not self.physics:
            self.X = Xs
            K = [matern52(Xs, Xs, h['ls'], h['sf2']) for h in self.hyp]
            self.alpha = [np.linalg.solve(K[o] + self.hyp[o]['noise'] * np.eye(n),
                                          self.Yn[o]) for o in range(NOUT)]
            return self
        # ---- support set = training inputs + collocation inputs -----------
        Z = np.vstack([Xs, Xc_s])
        m = len(Xc_s)
        N = n + m
        self.X = Z
        KZ = [matern52(Z, Z, h['ls'], h['sf2']) for h in self.hyp]
        E = np.zeros((n, N)); E[np.arange(n), np.arange(n)] = 1.0          # training rows
        F = np.zeros((m, N)); F[np.arange(m), np.arange(n, N)] = 1.0       # collocation rows
        A = []
        rhs0 = np.zeros(N * NOUT)
        for o in range(NOUT):
            s2 = self.hyp[o]['noise']
            A.append(E.T @ (E @ KZ[o]) / s2 + np.eye(N))
            rhs0[o * N:(o + 1) * N] = (E.T @ self.Yn[o]) / s2
        # constraint rows: c = (sum_o s_o*sd_o*g_o(x_c) + sum_o s_o*mu_o - B_in)/B_in
        G = np.hstack([(S_VEC[o] * self.hyp[o]['sd'] / Bin_c)[:, None] * (F @ KZ[o])
                       for o in range(NOUT)])
        b = (Bin_c - sum(S_VEC[o] * self.hyp[o]['mu'] for o in range(NOUT))) / Bin_c
        Ablk = np.zeros((N * NOUT, N * NOUT))
        for o in range(NOUT):
            Ablk[o * N:(o + 1) * N, o * N:(o + 1) * N] = A[o]
        GtG = G.T @ G
        eta = np.zeros(m)
        mu = self.mu0
        for k in range(self.n_outer):
            M = Ablk + 0.5 * mu * GtG
            rhs = rhs0 - 0.5 * G.T @ (eta - mu * b)
            alpha = np.linalg.solve(M, rhs)
            c = G @ alpha - b
            self.history.append(dict(outer=k, mu=mu, cnorm=float(np.abs(c).max()),
                                     crms=float(np.sqrt((c ** 2).mean()))))
            if np.abs(c).max() < self.tol:
                break
            eta = eta + mu * c
            mu = min(self.rho * mu, self.mu_max)
        self.alpha = [alpha[o * N:(o + 1) * N] for o in range(NOUT)]
        return self

    def predict(self, Xs, chunk=500):
        out = np.empty((len(Xs), NOUT))
        for i in range(0, len(Xs), chunk):               # chunked: keeps memory bounded
            sl = slice(i, min(i + chunk, len(Xs)))
            for o in range(NOUT):
                h = self.hyp[o]
                Ks = matern52(Xs[sl], self.X, h['ls'], h['sf2'])
                out[sl, o] = h['mu'] + h['sd'] * (Ks @ self.alpha[o])
        return out


# ------------------------------------------------------------------- MLP
def elu(z):
    return np.where(z > 0, z, np.exp(np.minimum(z, 0)) - 1.0)


def delu(z):
    return np.where(z > 0, 1.0, np.exp(np.minimum(z, 0)))


class NNSurrogate:
    """MLP with two hidden layers of 64 ELU units.

    physics=True : output head 1 is softmax over (m+2)=5 channels scaled by B_in,
                   giving [B_el,B_heat,B_ash,B_L,B_D] (Eq. 5); head 2 gives S_gen.
    physics=False: linear outputs for the five targets, no penalty.
    """

    def __init__(self, d, physics=True, penalty=None, hidden=64, seed=0, lr=3e-3, epochs=4000,
                 rho=1.6, mu0=1.0, mu_max=1e6, outer_every=160):
        rng = np.random.default_rng(seed)
        self.physics = physics
        # penalty defaults to on whenever a constrained variant is requested
        self.penalty = physics if penalty is None else penalty
        h = hidden
        self.p = dict(
            W1=rng.normal(0, np.sqrt(2 / d), (d, h)), b1=np.zeros(h),
            W2=rng.normal(0, np.sqrt(2 / h), (h, h)), b2=np.zeros(h),
            W3=rng.normal(0, np.sqrt(2 / h), (h, 5)), b3=np.zeros(5),
            W4=rng.normal(0, np.sqrt(2 / h), (h, 1)), b4=np.zeros(1))
        self.lr, self.epochs = lr, epochs
        self.rho, self.mu0, self.mu_max, self.outer_every = rho, mu0, mu_max, outer_every
        self.history = []

    # ---- forward -------------------------------------------------------
    def _forward(self, Xs, Bin):
        p = self.p
        z1 = Xs @ p['W1'] + p['b1']; a1 = elu(z1)
        z2 = a1 @ p['W2'] + p['b2']; a2 = elu(z2)
        z3 = a2 @ p['W3'] + p['b3']
        z4 = a2 @ p['W4'] + p['b4']
        cache = dict(Xs=Xs, z1=z1, a1=a1, z2=z2, a2=a2, z3=z3, z4=z4, Bin=Bin)
        if self.physics:
            e = np.exp(z3 - z3.max(1, keepdims=True))
            sm = e / e.sum(1, keepdims=True)
            flows = Bin[:, None] * sm                       # B_el,B_heat,B_ash,B_L,B_D
            sgen = self.sg_mu + self.sg_sd * z4[:, 0]
            cache.update(sm=sm, flows=flows, sgen=sgen)
            pred = np.column_stack([flows[:, :4], sgen])
        else:
            pred = z3 * self.y_sd + self.y_mu
            cache.update(pred=pred)
        return pred, cache

    def _residual(self, pred, cache):
        if self.physics:
            return (cache['flows'][:, 4] - T0 * cache['sgen'] / 1e3) / cache['Bin']
        return (cache['Bin'] - pred @ S_VEC) / cache['Bin']

    # ---- training ------------------------------------------------------
    def fit(self, Xs, Y, Bin, Xc_s=None, Bin_c=None):
        self.y_mu, self.y_sd = Y.mean(0), Y.std(0)
        self.sg_mu, self.sg_sd = Y[:, 4].mean(), Y[:, 4].std()
        if self.physics:
            # initialise the output bias at the mean exergy shares of the
            # training set, so the softmax starts near the operating point
            BD = Bin - Y[:, :4].sum(1)
            shares = np.column_stack([Y[:, :4], BD]).mean(0) / Bin.mean()
            self.p['b3'] = np.log(np.maximum(shares, 1e-8))
            self.p['b3'] -= self.p['b3'].mean()
        p = self.p
        mom = {k: np.zeros_like(v) for k, v in p.items()}
        vel = {k: np.zeros_like(v) for k, v in p.items()}
        eta_c = 0.0
        mu = self.mu0
        b1_, b2_, eps = 0.9, 0.999, 1e-8
        Yn = (Y - self.y_mu) / self.y_sd
        for it in range(1, self.epochs + 1):
            pred, cache = self._forward(Xs, Bin)
            resid_data = (pred - self.y_mu) / self.y_sd - Yn
            n = len(Xs)
            dpred = (2.0 / (n * 5)) * resid_data / self.y_sd
            loss_data = float((resid_data ** 2).mean())
            # physics term on training + collocation inputs
            cvec = self._residual(pred, cache)
            dL_dc = (eta_c + mu * cvec) / n if self.penalty else np.zeros(n)
            grads = self._backward(cache, dpred, dL_dc)
            if self.penalty and Xc_s is not None:
                predc, cachec = self._forward(Xc_s, Bin_c)
                cc = self._residual(predc, cachec)
                gc = self._backward(cachec, np.zeros_like(predc), (eta_c + mu * cc) / len(cc))
                for k in grads:
                    grads[k] += gc[k]
                cfull = np.concatenate([cvec, cc])
            else:
                cfull = cvec
            lr_t = self.lr * (0.5 * (1 + np.cos(np.pi * it / self.epochs)) * 0.98 + 0.02)
            for k in p:
                mom[k] = b1_ * mom[k] + (1 - b1_) * grads[k]
                vel[k] = b2_ * vel[k] + (1 - b2_) * grads[k] ** 2
                mh = mom[k] / (1 - b1_ ** it); vh = vel[k] / (1 - b2_ ** it)
                p[k] -= lr_t * mh / (np.sqrt(vh) + eps)
            if self.penalty and it % self.outer_every == 0:
                eta_c = eta_c + mu * float(np.mean(cfull))
                mu = min(self.rho * mu, self.mu_max)
            if it % 20 == 0 or it == 1:
                self.history.append(dict(epoch=it, loss_data=loss_data,
                                         loss_phy=float((cfull ** 2).mean()),
                                         mu=mu, cnorm=float(np.abs(cfull).max())))
        return self

    def _backward(self, cache, dpred, dL_dc):
        """dpred: dLoss/dpred (n,5); dL_dc: dLoss/dresidual (n,)"""
        p = self.p
        n = len(cache['Xs'])
        dz3 = np.zeros((n, 5)); dz4 = np.zeros((n, 1))
        if self.physics:
            sm, Bin, sgen = cache['sm'], cache['Bin'], cache['sgen']
            # via pred[:, :4] = Bin*sm[:, :4]
            dflow = np.zeros((n, 5))
            dflow[:, :4] += dpred[:, :4] * Bin[:, None]
            dsg = dpred[:, 4]
            # residual r = (Bin*sm4 - T0*sgen/1e3)/Bin
            dflow[:, 4] += dL_dc * 1.0 * Bin / Bin          # d r / d(Bin*sm4) = 1/Bin -> times Bin
            dsg = dsg + dL_dc * (-T0 / 1e3) / Bin
            # softmax jacobian: d(sm)/d(z3)
            dsm = dflow * Bin[:, None]
            dsm[:, 4] = dflow[:, 4]                          # already in flow units
            # recompute cleanly: dLoss/dsm_k = dLoss/dflow_k * Bin
            dsm = np.zeros((n, 5))
            dsm[:, :4] = dpred[:, :4] * Bin[:, None]
            dsm[:, 4] = dL_dc                                # d r/d sm4 = Bin/Bin = 1
            dz3 = sm * (dsm - (dsm * sm).sum(1, keepdims=True))
            dz4[:, 0] = dsg * self.sg_sd                     # sgen = sg_mu + sg_sd*z4
        else:
            dz3 = dpred * self.y_sd + (-dL_dc[:, None] * S_VEC[None, :] / cache['Bin'][:, None]) * self.y_sd
        g = {}
        g['W3'] = cache['a2'].T @ dz3; g['b3'] = dz3.sum(0)
        g['W4'] = cache['a2'].T @ dz4; g['b4'] = dz4.sum(0)
        da2 = dz3 @ p['W3'].T + dz4 @ p['W4'].T
        dz2 = da2 * delu(cache['z2'])
        g['W2'] = cache['a1'].T @ dz2; g['b2'] = dz2.sum(0)
        da1 = dz2 @ p['W2'].T
        dz1 = da1 * delu(cache['z1'])
        g['W1'] = cache['Xs'].T @ dz1; g['b1'] = dz1.sum(0)
        return g

    def predict(self, Xs, Bin):
        pred, _ = self._forward(Xs, Bin)
        return pred


# ---------------------------------------------------------------- helpers
def balance_residual(pred, Bin):
    """Normalised exergy-balance residual r_B (Eq. 6) for predictions in physical units."""
    return (Bin - pred @ S_VEC) / Bin


def destruction(pred, Bin):
    return Bin - pred[:, :4].sum(1)


def allocation(pred):
    P = pred[:, :3]
    return P / P.sum(1, keepdims=True)
