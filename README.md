# Bayesian Machine Learning

A six-week, university-level curriculum that derives modern Bayesian machine learning from first principles. Every probabilistic model, inference algorithm, and approximation scheme is rebuilt in pure NumPy / SciPy before being benchmarked against reference implementations (`PyMC`, `NumPyro`, `GPyTorch`, `Pyro`).

The course is presented as a sequence of self-contained Jupyter notebooks. Each notebook combines:

- **Theoretical background** — derivations from probability axioms, exponential-family identities, and information-theoretic principles.
- **Python implementations** — every algorithm coded from scratch, then compared with established libraries.
- **Visualizations** — posterior densities, MCMC traces, ELBO trajectories, GP credible bands, and BNN epistemic vs aleatoric uncertainty decomposition.
- **Exercises** — analytical derivations and computational extensions.

## Curriculum

| Week | Title | Core topics |
|------|-------|-------------|
| 1 | Foundations of Bayesian Inference | Bayes' theorem, prior/likelihood/posterior, conjugate families, exponential family, posterior predictive |
| 2 | Bayesian Linear Models | Bayesian linear regression, evidence approximation, ARD, Bayesian logistic regression via Laplace |
| 3 | Approximate Inference I — MCMC | Metropolis–Hastings, Gibbs, HMC, NUTS, diagnostics ($\hat{R}$, ESS) |
| 4 | Approximate Inference II — Variational Inference | ELBO, mean-field VI, CAVI, reparameterization trick, normalizing flows |
| 5 | Gaussian Processes | GP regression, kernels, hyperparameter learning, sparse GPs, classification |
| 6 | Bayesian Deep Learning | Bayesian neural networks, MC Dropout, Deep Ensembles, BNN via VI, epistemic vs aleatoric uncertainty |

## Repository structure

```
bayesian_machine_learning/
├── README.md
├── requirements.txt
├── LICENSE
├── week1_foundations/
│   ├── 01_bayes_theorem_from_scratch.ipynb
│   ├── 02_conjugate_priors.ipynb
│   └── 03_exponential_family_and_posterior_predictive.ipynb
├── week2_bayesian_linear_models/
│   ├── 01_bayesian_linear_regression.ipynb
│   ├── 02_evidence_approximation_and_ard.ipynb
│   └── 03_bayesian_logistic_regression_laplace.ipynb
├── week3_mcmc/
│   ├── 01_metropolis_hastings_from_scratch.ipynb
│   ├── 02_gibbs_sampling.ipynb
│   └── 03_hamiltonian_monte_carlo.ipynb
├── week4_variational_inference/
│   ├── 01_elbo_and_mean_field_vi.ipynb
│   ├── 02_cavi_for_gaussian_mixture.ipynb
│   └── 03_reparameterization_and_bbvi.ipynb
├── week5_gaussian_processes/
│   ├── 01_gp_regression_from_scratch.ipynb
│   ├── 02_kernels_and_hyperparameter_learning.ipynb
│   └── 03_sparse_gp_and_classification.ipynb
└── week6_bayesian_deep_learning/
    ├── 01_bayesian_neural_networks_via_vi.ipynb
    ├── 02_mc_dropout_and_deep_ensembles.ipynb
    └── 03_uncertainty_decomposition_and_calibration.ipynb
```

## Prerequisites

- Solid grounding in multivariate calculus and linear algebra
- Familiarity with probability theory (random variables, expectation, conditional distributions)
- Comfortable with Python, NumPy, and SciPy
- Exposure to PyTorch is helpful for weeks 4 and 6 but not required

## Installation

```bash
git clone https://github.com/HAYDARKILIC/bayesian_machine_learning.git
cd bayesian_machine_learning
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

## Course philosophy

There are excellent black-box probabilistic programming libraries (`PyMC`, `NumPyro`, `Stan`). This course is not built around them. The goal is to remove the magic: by the end of week 6, every algorithm a high-level library exposes — HMC, NUTS, mean-field VI, GP marginal-likelihood optimisation, BNN training via VI — should be code you have written yourself, debugged, and compared against the reference implementation. Libraries enter only after the from-scratch version exists.

## License

MIT. See `LICENSE`.
