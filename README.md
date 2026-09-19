# Physics-informed surrogates for exergy-based allocation: a waste-to-energy benchmark

Reference implementation and complete experiment code for the article

> Kant, N., Khera, R., Singh, P., Prakash, R., Ranganath, M. S., Singh, P. and Arora, A.
> *Physics-informed surrogates for exergy-based allocation in hybrid exergy–LCA frameworks:
> a reproducible waste-to-energy benchmark.* International Journal of Exergy (under review).

Every number, table and figure in the article is produced by this code.

## What is here

| File | Contents |
|---|---|
| `model.py` | Reference exergy model of a 500 t/day mass-burn waste-to-energy plant, with the input sampler (22 uncertain inputs, Dirichlet composition, Gaussian copula) |
| `surrogates.py` | Gaussian-process and neural-network surrogates, augmented-Lagrangian training, validator helpers |
| `runner.py` | Staged, resumable experiment: Monte Carlo, repetitions, learning curves |
| `run2.py` | Sobol' sensitivity analysis, tornado diagrams, adaptive sampling, validator fallback, timings |
| `figures.py` | Figures 1–7 |
| `results.json` | All reported numbers, as generated |

## Installation

```bash
pip install -r requirements.txt
```

Pure NumPy/SciPy/scikit-learn; no GPU required. The full study runs on a single CPU core.

## Reproducing the study

```bash
python runner.py setup            # Monte Carlo (10,000 samples) + GP hyperparameters
python runner.py main 0 8         # 8 repetitions of all five surrogate variants
python runner.py curve 60         # learning curve points
python runner.py curve 120
python runner.py curve 240
python runner.py curve 480
python runner.py finish           # aggregate -> results_part1.json, arrays_part1.npz
python run2.py                    # sensitivity, fallback, timings -> results.json
python figures.py                 # Figures 1-7
```

Each stage checkpoints to `ck/`, so the run can be resumed.

## The model

Steady-state exergy balance including destruction:

    B_in = sum_j B_P,j + B_L + B_D,    B_D = T0 * S_gen >= 0

Exergy destruction is computed component-wise (combustion, boiler, power block); the
balance is **not** used to infer destruction by difference. Closure is exact to within
3e-16 of `B_in` at every sampled point — verify with:

```python
import numpy as np, model as M
X = M.sample_inputs(1000, np.random.default_rng(0))
print(max(abs(M.reference_model(x)['closure']) for x in X))
```

Co-products are net electricity, district heat and bottom ash; allocation factors are
`lambda_j = B_P,j / sum_k B_P,k`.

## Scope and limitations

This is a **documented reference benchmark built from representative composition and
operating data, not measurements from a specific facility.** It is calibrated against
published European plant statistics: its mixture heating value of 11.17 MJ/kg is close to
the 2.9 MWh/tonne (10.4 MJ/kg) average used in the EU R1 guidance, and its net electrical
efficiency of 20.6% at 182 kt/year lies within the reported range for European plants.
It is a reduced-order model: results should not be read as performance figures for any
operating plant.

## Main results

- GP surrogate reproduces the Monte Carlo allocation distributions (Kolmogorov–Smirnov
  p >= 0.96) with 0.09 percentage-point mean error from 440 reference evaluations.
- The physics penalty does not improve accuracy, but cuts network validator rejections
  from 51.6% to 37.2%.
- Enforcing closure in the network output layer (softmax over channels including
  destruction) degrades accuracy about fourfold, because destruction is ~72% of input
  exergy.

## License

MIT (see `LICENSE`).
