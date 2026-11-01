# MRTT Bayesian trust-update model

Hierarchical Bayesian model of the human investor in the multi-round trust task (MRTT), fit with Stan via `cmdstanpy`.

- `trust_update.stan`: the model. The investor holds a Beta belief about the fraction of the tripled amount the trustee returns, updates it after each round (weight `eta`), and picks an investment 0-20 with a softmax over a mean-variance utility (`beta` temperature, `rho` risk aversion, `mu0` prior mean).
- `fit.py <RND|MAX|FAIR>`: fits one condition, prints diagnostics, LOO, and a posterior-predictive check.
- `data/human_mrtt.csv`: all human data. Columns: `cond, pid, t, investment, k`. `k` is the trustee's repayment level 0-4; repayment = `0.75 * k * investment`.

## Setup (cluster)

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"   # needs a C++ toolchain (g++ / make)
python fit.py RND
```

If CmdStan lives elsewhere, set `CMDSTAN_DIR=/path/to/cmdstan-x.y.z`.

SLURM: `mkdir -p slurm/logs && sbatch slurm/fit.sbatch` runs one array task per condition.

## Note

The data are human participant data. Keep the repository private.
