"""Generate Figures 1-7 from the measured results."""
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                     'axes.titlesize': 10, 'axes.labelsize': 9,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 600, 'savefig.dpi': 600, 'legend.frameon': False})
C = dict(mc='#4C72B0', gp='#DD8452', nn='#55A868', plain='#8C8C8C', hard='#C44E52',
         soft='#937860')
res = json.load(open('results.json'))
a1 = dict(np.load('arrays_part1.npz'))
hist = json.load(open('hist.json'))
OUT = '/home/claude/build/media/'
VAR = ['GP-PI', 'GP-plain', 'NN-hard', 'NN-soft', 'NN-plain']
VC = {'GP-PI': C['gp'], 'GP-plain': C['plain'], 'NN-hard': C['hard'],
      'NN-soft': C['nn'], 'NN-plain': C['soft']}


def lab(ax, s):
    ax.text(-0.14, 1.06, s, transform=ax.transAxes, fontweight='bold', fontsize=11)


# ------------------------------------------------------------------ Figure 1
fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.4))
r = np.linspace(0, 5, 400)
for ls, c in zip([0.5, 1.0, 2.0], ['#C44E52', '#DD8452', '#4C72B0']):
    rr = r / ls
    ax[0, 0].plot(r, (1 + np.sqrt(5) * rr + 5 / 3 * rr ** 2) * np.exp(-np.sqrt(5) * rr),
                  color=c, label=r'$\ell$ = %.1f' % ls)
ax[0, 0].set_xlabel('distance $r$'); ax[0, 0].set_ylabel('$k_{5/2}(r)$')
ax[0, 0].set_title('Matérn-5/2 covariance'); ax[0, 0].legend()
lab(ax[0, 0], '(a)')

# NN schematic
s = ax[0, 1]; s.axis('off'); s.set_xlim(0, 10); s.set_ylim(0, 10)
for j, (xx, n, t) in enumerate([(1.0, 4, 'inputs\n$\\mathbf{x}$'), (3.4, 5, '64 ELU'),
                                (5.8, 5, '64 ELU')]):
    for k in range(n):
        s.add_patch(plt.Circle((xx, 8.2 - 1.35 * k), 0.26, fc='white', ec='#555555', lw=.8))
    s.text(xx, 1.9, t, ha='center', fontsize=8)
for k in range(5):
    s.add_patch(plt.Circle((8.3, 8.6 - 1.05 * k), 0.24, fc='#EAEAEA', ec='#555555', lw=.8))
s.text(8.3, 2.6, 'softmax $\\times\\, B_{in}$\n$[B_{P,j}, B_L, B_D]$', ha='center', fontsize=8)
s.add_patch(plt.Circle((8.3, 1.4), 0.24, fc='#EAEAEA', ec='#555555', lw=.8))
s.text(9.9, 1.4, '$\\hat{S}_{gen}$', fontsize=9, va='center')
for x0, x1, n0, n1 in [(1.0, 3.4, 4, 5), (3.4, 5.8, 5, 5)]:
    for i in range(n0):
        for j in range(n1):
            s.plot([x0 + .26, x1 - .26], [8.2 - 1.35 * i, 8.2 - 1.35 * j],
                   color='#CCCCCC', lw=.3, zorder=0)
for j in range(5):
    s.plot([6.06, 8.06], [8.2 - 1.35 * 2, 8.6 - 1.05 * j], color='#CCCCCC', lw=.3, zorder=0)
s.plot([6.06, 8.06], [8.2 - 1.35 * 3, 1.4], color='#CCCCCC', lw=.3, zorder=0)
s.set_title('Constrained network (Eq. 5)')
lab(s, '(b)')

h = hist['nn']
ep = [d['epoch'] for d in h]
ax[1, 0].semilogy(ep, [d['loss_data'] for d in h], color=C['mc'], label='$\\mathcal{L}_{data}$')
ax[1, 0].semilogy(ep, [max(d['loss_phy'], 1e-16) for d in h], '--', color=C['hard'],
                  label='mean $r_B^2$')
ax[1, 0].set_xlabel('epoch'); ax[1, 0].set_ylabel('loss'); ax[1, 0].legend()
ax[1, 0].set_title('Network training')
lab(ax[1, 0], '(c)')

al = hist['al']
k = [d['outer'] for d in al]
axa = ax[1, 1]
axa.semilogy(k, [d['mu'] for d in al], 'o-', ms=3, color='#C44E52', label='$\\mu^k$')
axb = axa.twinx(); axb.spines['top'].set_visible(False)
axb.semilogy(k, [max(d['cnorm'], 1e-12) for d in al], 's-', ms=3, color=C['nn'],
             label='$\\|c(\\theta)\\|_\\infty$')
axa.set_xlabel('outer iteration $k$'); axa.set_ylabel('$\\mu^k$', color='#C44E52')
axb.set_ylabel('$\\|c\\|_\\infty$', color=C['nn'])
axa.set_title('Augmented-Lagrangian iterations')
lab(axa, '(d)')
fig.tight_layout(); fig.savefig(OUT + 'fig1.png'); plt.close(fig)

# ------------------------------------------------------------------ Figure 2
fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.4))
from scipy.stats import qmc
rng = np.random.default_rng(4)
lhs = qmc.LatinHypercube(d=2, seed=4).random(40); rnd = rng.random((40, 2))
ax[0, 0].scatter(rnd[:, 0], rnd[:, 1], s=16, color=C['plain'], label='random')
ax[0, 0].scatter(lhs[:, 0], lhs[:, 1], s=16, color=C['gp'], marker='^', label='LHS')
for t in np.linspace(0, 1, 9):
    ax[0, 0].axvline(t, lw=.3, color='#DDDDDD'); ax[0, 0].axhline(t, lw=.3, color='#DDDDDD')
ax[0, 0].set_xlabel('$x_1$ (normalised)'); ax[0, 0].set_ylabel('$x_2$ (normalised)')
ax[0, 0].legend(loc='upper center', ncol=2, bbox_to_anchor=(.5, 1.02))
ax[0, 0].set_title('Space-filling design')
lab(ax[0, 0], '(a)')

g = res['gamma']
names = sorted(g, key=lambda k: -g[k])[:8]
ax[0, 1].barh(range(len(names))[::-1], [g[n] for n in names], color=C['gp'])
ax[0, 1].set_yticks(range(len(names))[::-1]); ax[0, 1].set_yticklabels(names, fontsize=7)
ax[0, 1].set_xlabel('$\\gamma_p$'); ax[0, 1].set_title('Importance scores (Eq. 9)')
lab(ax[0, 1], '(b)')

act = np.array(res['active']['active']); lhsc = np.array(res['active']['lhs'])
b = res['active']['batches']; sd = res['active']['sd']
axa = ax[1, 0]
axa.bar(range(len(b)), b, color=C['mc'], alpha=.75)
axa.set_xlabel('active-learning iteration'); axa.set_ylabel('batch size $N_{batch}$', color=C['mc'])
axb = axa.twinx(); axb.spines['top'].set_visible(False)
axb.plot(range(len(sd)), sd, 'o-', ms=3, color=C['hard'])
axb.set_ylabel('max predictive sd (MW)', color=C['hard'])
axa.set_title('Adaptive batch sizing')
lab(axa, '(c)')

ax[1, 1].plot(act[:, 0], act[:, 1], 'o-', ms=3, color=C['gp'], label='adaptive')
ax[1, 1].plot(lhsc[:, 0], lhsc[:, 1], 's--', ms=3, color=C['plain'], label='LHS only')
ax[1, 1].set_yscale('log'); ax[1, 1].set_xlabel('reference-model evaluations')
ax[1, 1].set_ylabel('MAE of $\\lambda$ (pp)'); ax[1, 1].legend()
ax[1, 1].set_title('Adaptive vs space-filling')
lab(ax[1, 1], '(d)')
fig.tight_layout(); fig.savefig(OUT + 'fig2.png'); plt.close(fig)

# ------------------------------------------------------------------ Figure 3
lam_mc = a1['lam_mc']; lam_gp = a1['lam_GP-PI']; lam_nn = a1['lam_NN-soft']
fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.4))
titles = ['Electricity, $\\lambda_{el}$', 'District heat, $\\lambda_{heat}$',
          'Bottom ash, $\\lambda_{ash}$']
for j, axx in enumerate([ax[0, 0], ax[0, 1], ax[1, 0]]):
    bins = np.linspace(min(lam_mc[:, j].min(), lam_gp[:, j].min()),
                       max(lam_mc[:, j].max(), lam_gp[:, j].max()), 60)
    axx.hist(lam_mc[:, j], bins=bins, density=True, color=C['mc'], alpha=.55,
             label='Monte Carlo')
    axx.hist(lam_gp[:, j], bins=bins, density=True, histtype='step', lw=1.4,
             color=C['gp'], label='GP-PI')
    axx.hist(lam_nn[:, j], bins=bins, density=True, histtype='step', lw=1.1,
             color=C['nn'], ls='--', label='NN-soft')
    axx.set_xlabel('$\\lambda$'); axx.set_ylabel('density'); axx.set_title(titles[j])
    if j == 0:
        axx.legend(fontsize=7)
    lab(axx, '(%s)' % 'abc'[j])
ax[1, 1].scatter(lam_mc[:, 0], lam_mc[:, 1], s=2, alpha=.15, color=C['mc'], label='Monte Carlo')
ax[1, 1].scatter(lam_gp[:, 0], lam_gp[:, 1], s=2, alpha=.15, color=C['gp'], label='GP-PI')
rho_mc = res['mc']['corr'][0][1]; rho_gp = np.corrcoef(lam_gp[:, 0], lam_gp[:, 1])[0, 1]
ax[1, 1].set_xlabel('$\\lambda_{el}$'); ax[1, 1].set_ylabel('$\\lambda_{heat}$')
ax[1, 1].set_title('Joint distribution ($\\rho$ = %.3f vs %.3f)' % (rho_mc, rho_gp))
ax[1, 1].legend(fontsize=7, markerscale=4)
lab(ax[1, 1], '(d)')
fig.tight_layout(); fig.savefig(OUT + 'fig3.png'); plt.close(fig)

# ------------------------------------------------------------------ Figure 4
import surrogates as S
Yte, Bte = a1['Yte'], a1['Bte']
lref = Yte[:, :3] / Yte[:, :3].sum(1, keepdims=True)
fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.4))
for axx, v, t in [(ax[0, 0], 'GP-PI', 'GP-PI'), (ax[0, 1], 'NN-soft', 'NN-soft')]:
    p = a1['predte_' + v]; lam = p[:, :3] / p[:, :3].sum(1, keepdims=True)
    axx.scatter(lref[:, 0], lam[:, 0], s=8, alpha=.6, color=VC[v])
    lo, hi = lref[:, 0].min(), lref[:, 0].max()
    axx.plot([lo, hi], [lo, hi], 'k-', lw=.8)
    r2 = 1 - ((lam[:, 0] - lref[:, 0]) ** 2).sum() / ((lref[:, 0] - lref[:, 0].mean()) ** 2).sum()
    axx.set_xlabel('reference $\\lambda_{el}$'); axx.set_ylabel('predicted $\\lambda_{el}$')
    axx.set_title('%s ($R^2$ = %.4f)' % (t, r2))
lab(ax[0, 0], '(a)'); lab(ax[0, 1], '(b)')
for v in ['GP-PI', 'NN-soft', 'NN-hard']:
    p = a1['predte_' + v]; lam = p[:, :3] / p[:, :3].sum(1, keepdims=True)
    ax[1, 0].hist(100 * (lam[:, 0] - lref[:, 0]), bins=40, histtype='step', lw=1.2,
                  color=VC[v], label=v)
ax[1, 0].set_xlabel('$\\lambda_{el}$ residual (pp)'); ax[1, 0].set_ylabel('count')
ax[1, 0].legend(fontsize=7); ax[1, 0].set_title('Residual distribution')
lab(ax[1, 0], '(c)')
for v in VAR:
    ns = sorted(int(k) for k in res['curve'][v])
    m = [res['curve'][v][str(n)]['mean'] for n in ns]
    ci = [res['curve'][v][str(n)]['ci'] for n in ns]
    ax[1, 1].errorbar(ns, m, yerr=ci, marker='o', ms=3, lw=1.1, capsize=2,
                      color=VC[v], label=v)
ax[1, 1].set_xscale('log'); ax[1, 1].set_yscale('log')
ax[1, 1].set_xlabel('training-set size'); ax[1, 1].set_ylabel('MAE of $\\lambda$ (pp)')
ax[1, 1].legend(fontsize=7); ax[1, 1].set_title('Learning curves (mean, 95% CI)')
lab(ax[1, 1], '(d)')
fig.tight_layout(); fig.savefig(OUT + 'fig4.png'); plt.close(fig)

# ------------------------------------------------------------------ Figure 5
fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.4))
sw = res['tornado']['swings']
for j, (axx, name) in enumerate([(ax[0, 0], '$\\lambda_{el}$'), (ax[0, 1], '$\\lambda_{heat}$')]):
    keys = sorted(sw, key=lambda k: -max(abs(sw[k]['hi'][j]), abs(sw[k]['lo'][j])))[:8]
    y = np.arange(len(keys))[::-1]
    axx.barh(y, [sw[k]['hi'][j] for k in keys], color=C['gp'], height=.6, label='+1 sd')
    axx.barh(y, [sw[k]['lo'][j] for k in keys], color=C['mc'], height=.6, label='-1 sd')
    axx.set_yticks(y); axx.set_yticklabels(keys, fontsize=7)
    axx.axvline(0, color='k', lw=.8)
    axx.set_xlabel('change in %s (pp)' % name)
    axx.set_title('Tornado, %s' % name)
    if j == 0:
        axx.legend(fontsize=7)
lab(ax[0, 0], '(a)'); lab(ax[0, 1], '(b)')
S1, ST = res['sobol']['S1'], res['sobol']['ST']
keys = sorted(S1, key=lambda k: -S1[k][0])[:7]
xx = np.arange(len(keys)); w = .26
for j, (axx, d, t) in enumerate([(ax[1, 0], S1, 'First-order $S_1$'),
                                 (ax[1, 1], ST, 'Total-order $S_T$')]):
    for c, (jj, nm) in enumerate([(0, '$\\lambda_{el}$'), (1, '$\\lambda_{heat}$'),
                                  (2, '$\\lambda_{ash}$')]):
        axx.bar(xx + (c - 1) * w, [d[k][jj] for k in keys], width=w, label=nm,
                color=[C['mc'], C['gp'], C['nn']][c])
    axx.set_xticks(xx); axx.set_xticklabels(keys, rotation=35, ha='right', fontsize=6.5)
    axx.set_ylabel('index'); axx.set_title(t)
    if j == 0:
        axx.legend(fontsize=7)
lab(ax[1, 0], '(c)'); lab(ax[1, 1], '(d)')
fig.tight_layout(); fig.savefig(OUT + 'fig5.png'); plt.close(fig)

# ------------------------------------------------------------------ Figure 6
fb = res['fallback']
fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.4))
for v in ['GP-PI', 'NN-hard', 'NN-soft', 'NN-plain']:
    r = np.abs(a1['resid_' + v])
    ax[0, 0].hist(np.log10(np.maximum(r, 1e-12)), bins=60, histtype='step', lw=1.2,
                  color=VC[v], label=v)
ax[0, 0].axvline(np.log10(1e-2), color='k', ls='--', lw=.9)
ax[0, 0].text(np.log10(1e-2), ax[0, 0].get_ylim()[1] * .9, ' $\\varepsilon$', fontsize=8)
ax[0, 0].set_xlabel('$\\log_{10}|r_B|$'); ax[0, 0].set_ylabel('count')
ax[0, 0].legend(fontsize=7); ax[0, 0].set_title('Exergy-balance residual (Eq. 6)')
lab(ax[0, 0], '(a)')
for v in VAR:
    d = np.array(fb[v]['rej_vs_eps'])
    ax[0, 1].loglog(d[:, 0], np.maximum(d[:, 1], 1e-3), '-', lw=1.2, color=VC[v], label=v)
ax[0, 1].axvline(1e-2, color='k', ls='--', lw=.9)
ax[0, 1].set_xlabel('tolerance $\\varepsilon$'); ax[0, 1].set_ylabel('rejection rate (%)')
ax[0, 1].legend(fontsize=7); ax[0, 1].set_title('Rejections versus tolerance')
lab(ax[0, 1], '(b)')
xx = np.arange(len(VAR))
ax[1, 0].bar(xx - .2, [fb[v]['mae_before'] for v in VAR], width=.4, color=C['plain'],
             label='surrogate only')
ax[1, 0].bar(xx + .2, [fb[v]['mae_after'] for v in VAR], width=.4, color=C['gp'],
             label='with fallback')
ax[1, 0].set_xticks(xx); ax[1, 0].set_xticklabels(VAR, rotation=20, fontsize=7)
ax[1, 0].set_yscale('log'); ax[1, 0].set_ylabel('MAE of $\\lambda$ (pp)')
ax[1, 0].legend(fontsize=7); ax[1, 0].set_title('Effect of reference-model fallback')
lab(ax[1, 0], '(c)')
ax[1, 1].bar(xx, [fb[v]['rej_rate'] for v in VAR], color=[VC[v] for v in VAR])
ax[1, 1].set_xticks(xx); ax[1, 1].set_xticklabels(VAR, rotation=20, fontsize=7)
ax[1, 1].set_ylabel('rejection rate at $\\varepsilon = 10^{-2}$ (%)')
ax[1, 1].set_title('Validator activity')
lab(ax[1, 1], '(d)')
fig.tight_layout(); fig.savefig(OUT + 'fig6.png'); plt.close(fig)

# ------------------------------------------------------------------ Figure 7
tm = res['timing']
fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.4))
calls = dict(zip(VAR, [240 + 200 + int(fb[v]['rej_rate'] / 100 * 10000) for v in VAR]))
ax[0, 0].bar(['Monte\nCarlo'] + VAR, [10000] + [calls[v] for v in VAR],
             color=[C['mc']] + [VC[v] for v in VAR])
ax[0, 0].set_ylabel('reference-model evaluations')
ax[0, 0].tick_params(axis='x', labelsize=7, rotation=20)
ax[0, 0].set_title('Reference-model calls')
lab(ax[0, 0], '(a)')
ax[0, 1].bar(['reference', 'GP', 'NN'], [tm['t_ref_ms'], tm['t_gp_ms'], tm['t_nn_ms']],
             color=[C['mc'], C['gp'], C['nn']])
ax[0, 1].set_yscale('log'); ax[0, 1].set_ylabel('time per sample (ms)')
ax[0, 1].set_title('Measured evaluation cost')
lab(ax[0, 1], '(b)')
tref = np.logspace(-2, 3, 60)      # s per reference evaluation
for v, tsur, ttrain in [('GP-PI', tm['t_gp_ms'] / 1e3, res['main']['GP-PI']['tfit']['mean']),
                        ('NN-soft', tm['t_nn_ms'] / 1e3, res['main']['NN-soft']['tfit']['mean'])]:
    nc = calls[v]
    tot = nc * tref + ttrain + 10000 * tsur
    ax[1, 0].loglog(tref, 10000 * tref / tot, lw=1.4, color=VC[v], label=v)
ax[1, 0].axhline(1, color='k', lw=.8, ls=':')
ax[1, 0].axvline(tm['t_ref_ms'] / 1e3, color=C['plain'], ls='--', lw=.9)
ax[1, 0].text(tm['t_ref_ms'] / 1e3, 1.4, ' this study', fontsize=7, rotation=90)
ax[1, 0].set_xlabel('cost of one reference evaluation (s)')
ax[1, 0].set_ylabel('speed-up factor'); ax[1, 0].legend(fontsize=7)
ax[1, 0].set_title('Projected speed-up')
lab(ax[1, 0], '(c)')
ax[1, 1].bar(VAR, [res['main'][v]['tfit']['mean'] for v in VAR],
             yerr=[res['main'][v]['tfit']['sd'] for v in VAR],
             color=[VC[v] for v in VAR], capsize=3)
ax[1, 1].set_ylabel('training time (s)'); ax[1, 1].tick_params(axis='x', labelsize=7, rotation=20)
ax[1, 1].set_title('Surrogate training cost')
lab(ax[1, 1], '(d)')
fig.tight_layout(); fig.savefig(OUT + 'fig7.png'); plt.close(fig)
print('figures written')
