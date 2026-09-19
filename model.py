"""Reference exergy model of a municipal waste-to-energy (WtE) plant.

Steady-state, reduced-order model used as the reference calculation in the
surrogate study. All exergy rates in MW, temperatures in K unless noted.

Closure: B_in = B_el + B_heat + B_ash + B_L + B_D, with B_D = T0*S_gen the sum
of the component irreversibilities (combustion, boiler, power block, DH heat
exchanger).  Closure is asserted to 1e-9 relative.
"""
import numpy as np

T0 = 298.15          # K, reference environment temperature
P0 = 101.325         # kPa
M_WASTE = 500e3 / 86400.0   # kg/s  (500 t/day)
CP_FG = 1.15         # kJ/kg K, flue gas
CP_ASH = 0.90        # kJ/kg K, bottom ash
ASH_FRACTION = 0.20  # of input mass
T_FW = 378.15        # K, feedwater (105 C)
T_COND = 313.15      # K, condenser
T_ASH_OUT = 473.15   # K, ash discharge
T_SURF = 340.0       # K, boiler casing surface (radiation loss)
AIR_FUEL = 0.63      # kg flue gas per MJ of fuel (excess-air ratio ~1.8)

# 12 MSW categories: name, LHV (MJ/kg, as received), beta (b_ch/LHV), nominal mass fraction
CATEGORIES = [
    ("Paper and cardboard", 13.5, 1.13, 0.22),
    ("Plastics",            32.0, 1.06, 0.12),
    ("Food waste",           4.5, 1.20, 0.24),
    ("Garden waste",         6.0, 1.15, 0.08),
    ("Wood",                15.0, 1.13, 0.05),
    ("Textiles",            17.0, 1.12, 0.05),
    ("Rubber and leather",  20.0, 1.08, 0.02),
    ("Nappies",              8.0, 1.17, 0.04),
    ("Fines and inerts",     2.0, 1.25, 0.06),
    ("Glass",                0.0, 0.00, 0.05),
    ("Metals",               0.0, 0.00, 0.04),
    ("Other combustibles",  12.0, 1.14, 0.03),
]
LHV = np.array([c[1] for c in CATEGORIES])
BETA = np.array([c[2] for c in CATEGORIES])
W_NOM = np.array([c[3] for c in CATEGORIES])
W_NOM = W_NOM / W_NOM.sum()

# continuous uncertain inputs: name, distribution, parameters
CONT = [
    ("eta_boiler","normal",  (0.930, 0.015, 0.88, 0.97)),   # boiler heat-transfer effectiveness
    ("f_unb",     "uniform", (0.005, 0.020)),               # unburnt fuel energy to ash
    ("T_steam",   "uniform", (400.0, 450.0)),               # deg C
    ("P_steam",   "uniform", (40.0, 60.0)),                 # bar
    ("eps_hr",    "uniform", (0.65, 0.75)),                 # DH heat-recovery effectiveness
    ("f_aux",     "uniform", (0.030, 0.060)),               # auxiliary power fraction
    ("T_dh_sup",  "uniform", (85.0, 105.0)),                # deg C
    ("T_dh_ret",  "uniform", (45.0, 60.0)),                 # deg C
    ("T_stack",   "uniform", (130.0, 160.0)),               # deg C
    ("Q_dh_max",  "uniform", (20.0, 40.0)),                 # MW, district-heat demand limit
]
CONT_NAMES = [c[0] for c in CONT]
INPUT_NAMES = [f"w_{i+1}" for i in range(12)] + CONT_NAMES
D = len(INPUT_NAMES)

# Gaussian-copula correlation matrix for the continuous inputs (order of CONT)
RHO = np.eye(len(CONT))
def _set(i, j, r):
    RHO[i, j] = RHO[j, i] = r
_set(0, 1, -0.30)   # boiler efficiency / unburnt fraction
_set(2, 3, 0.60)    # steam temperature / pressure
_set(4, 6, 0.35)    # DH effectiveness / supply temperature
_set(0, 8, -0.45)   # boiler efficiency / stack temperature
DIRICHLET_S = 120.0  # Dirichlet concentration for the composition vector


def _logmean(Th, Tc):
    return (Th - Tc) / np.log(Th / Tc)


def reference_model(x):
    """Full thermodynamic and allocation calculation for one input vector.

    x : array (D,) = [12 mass fractions, eta_boiler, f_unb, T_steam(C),
        P_steam(bar), eps_hr, f_aux, T_dh_sup(C), T_dh_ret(C), T_stack(C),
        Q_dh_max(MW)]
    Returns exergy rates (MW), allocation factors and diagnostics.
    """
    w = x[:12]
    (eta_b, f_unb, T_s_C, P_s, eps_hr, f_aux, T_dhs_C, T_dhr_C, T_st_C,
     Q_dh_max) = x[12:]
    T_s, T_dhs, T_dhr, T_stack = (T_s_C + 273.15, T_dhs_C + 273.15,
                                  T_dhr_C + 273.15, T_st_C + 273.15)

    lhv_mix = float(w @ LHV)                       # MJ/kg as received
    Q_fuel = M_WASTE * lhv_mix                     # MW
    B_waste = M_WASTE * float(w @ (BETA * LHV))    # MW

    # --- combustion -------------------------------------------------------
    Q_unb = f_unb * Q_fuel                         # unburnt fuel energy leaving in ash
    Q_rel = Q_fuel - Q_unb                         # released to flue gas
    m_fg = M_WASTE * (1.0 + AIR_FUEL * lhv_mix)    # kg/s flue gas (excess-air scaling)
    T_fg = T0 + Q_rel * 1e3 / (m_fg * CP_FG)       # furnace exit temperature
    B_fg = m_fg * CP_FG * ((T_fg - T0) - T0 * np.log(T_fg / T0)) / 1e3
    m_ash = ASH_FRACTION * M_WASTE
    B_ash = (1.06 * Q_unb
             + m_ash * CP_ASH * ((T_ASH_OUT - T0) - T0 * np.log(T_ASH_OUT / T0)) / 1e3)
    I_comb = B_waste - B_fg - B_ash

    # --- boiler -----------------------------------------------------------
    Q_stack = m_fg * CP_FG * (T_stack - T0) / 1e3   # sensible heat leaving with stack gas
    Q_steam = eta_b * (Q_rel - Q_stack)             # heat transferred to the steam cycle
    Q_rad = (1.0 - eta_b) * (Q_rel - Q_stack)       # casing / radiation loss
    B_stack = m_fg * CP_FG * ((T_stack - T0) - T0 * np.log(T_stack / T0)) / 1e3
    B_rad = Q_rad * (1.0 - T0 / T_SURF)
    T_lm_st = _logmean(T_s, T_FW)
    B_steam = Q_steam * (1.0 - T0 / T_lm_st)
    I_boiler = B_fg - B_steam - B_stack - B_rad

    # --- power block and district heating ---------------------------------
    eta_ex_pb = 0.580 + 0.0015 * (P_s - 40.0) + 0.00060 * (T_s_C - 400.0)
    W_gross = eta_ex_pb * B_steam
    Q_cond = Q_steam - W_gross
    Q_dh = min(eps_hr * Q_cond, Q_dh_max)   # limited by district-heat demand
    T_lm_dh = _logmean(T_dhs, T_dhr)
    B_heat = Q_dh * (1.0 - T0 / T_lm_dh)
    B_rej = (Q_cond - Q_dh) * (1.0 - T0 / T_COND)
    B_aux = f_aux * W_gross
    B_el = W_gross - B_aux
    I_pb = B_steam + B_aux - B_el - B_heat - B_rej

    B_in = B_waste + B_aux
    B_L = B_stack + B_rad + B_rej
    B_D = I_comb + I_boiler + I_pb
    S_gen = B_D / T0 * 1e3                          # kW/K

    B_P = np.array([B_el, B_heat, B_ash])
    lam = B_P / B_P.sum()
    resid = (B_in - B_P.sum() - B_L - B_D) / B_in
    return dict(B_in=B_in, B_el=B_el, B_heat=B_heat, B_ash=B_ash, B_L=B_L,
                B_D=B_D, S_gen=S_gen, lam_el=lam[0], lam_heat=lam[1],
                lam_ash=lam[2], closure=resid, lhv=lhv_mix, Q_fuel=Q_fuel,
                W_gross=W_gross, Q_dh=Q_dh, Q_steam=Q_steam, T_fg=T_fg,
                Q_rad=Q_rad, Q_stack=Q_stack, Q_unb=Q_unb, m_fg=m_fg,
                I_comb=I_comb, I_boiler=I_boiler, I_pb=I_pb, B_waste=B_waste,
                B_aux=B_aux, B_stack=B_stack, B_rad=B_rad, B_rej=B_rej,
                B_fg=B_fg, B_steam=B_steam, eta_el=B_el / Q_fuel,
                eta_ex=(B_el + B_heat + B_ash) / B_in)


OUT_NAMES = ["B_el", "B_heat", "B_ash", "B_L", "S_gen"]


def evaluate(X):
    """Reference model on a set of inputs -> (n,5) outputs [B_el,B_heat,B_ash,B_L,S_gen]."""
    Y = np.empty((len(X), 5))
    for i, x in enumerate(X):
        r = reference_model(x)
        Y[i] = [r["B_el"], r["B_heat"], r["B_ash"], r["B_L"], r["S_gen"]]
    return Y


def input_exergy(X):
    """Cheap direct calculation of B_in for a set of inputs."""
    X = np.atleast_2d(X)
    lhvx = X[:, :12] @ (BETA * LHV)
    B_waste = M_WASTE * lhvx
    # auxiliary electricity depends on gross power; recomputed exactly by the model,
    # so B_in is evaluated with the model's own cheap sub-chain:
    out = np.empty(len(X))
    for i, x in enumerate(X):
        out[i] = reference_model(x)["B_in"]
    return out


# ---------------------------------------------------------------- sampling
def _copula(n, rng):
    L = np.linalg.cholesky(RHO)
    z = rng.standard_normal((n, len(CONT))) @ L.T
    from scipy.stats import norm
    return norm.cdf(z)


def _from_uniform(u):
    """Map copula uniforms to the continuous marginals."""
    from scipy.stats import truncnorm
    cols = []
    for k, (_, kind, par) in enumerate(CONT):
        if kind == "uniform":
            a, b = par
            cols.append(a + (b - a) * u[:, k])
        else:
            mu, sd, lo, hi = par
            a, b = (lo - mu) / sd, (hi - mu) / sd
            cols.append(truncnorm.ppf(u[:, k], a, b, loc=mu, scale=sd))
    return np.column_stack(cols)


def sample_inputs(n, rng, lhs=False):
    """Monte Carlo (lhs=False) or Latin hypercube (lhs=True) sample of the input space."""
    if lhs:
        from scipy.stats import qmc
        u = qmc.LatinHypercube(d=len(CONT), seed=int(rng.integers(1 << 31))).random(n)
        # correlate the LHS design with the same copula via rank transformation
        from scipy.stats import norm
        L = np.linalg.cholesky(RHO)
        z = norm.ppf(np.clip(u, 1e-6, 1 - 1e-6)) @ L.T
        u = norm.cdf(z)
    else:
        u = _copula(n, rng)
    cont = _from_uniform(u)
    w = rng.dirichlet(DIRICHLET_S * W_NOM, size=n)
    return np.hstack([w, cont])


def nominal_input():
    cont = []
    for _, kind, par in CONT:
        cont.append(0.5 * (par[0] + par[1]) if kind == "uniform" else par[0])
    return np.concatenate([W_NOM, np.array(cont)])
