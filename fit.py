import os
import sys

import arviz as az
import cmdstanpy
import numpy as np

from load_data import stan_data

HERE = os.path.dirname(os.path.abspath(__file__))
CMDSTAN = os.environ.get('CMDSTAN_DIR')
if CMDSTAN:
    cmdstanpy.set_cmdstan_path(CMDSTAN)

cond = sys.argv[1] if len(sys.argv) > 1 else 'RND'
data, pids = stan_data(cond)

model = cmdstanpy.CmdStanModel(stan_file=os.path.join(HERE, 'trust_update.stan'))
fit = model.sample(data=data, chains=4, parallel_chains=4, iter_warmup=1000, iter_sampling=1000,
                   seed=1010, adapt_delta=0.9, output_dir=os.path.join(HERE, f'fit_{cond}'))

print(fit.diagnose())
print(fit.summary(vars=['mu_g', 'sd_g']).round(3))

idata = az.from_cmdstanpy(fit, log_likelihood='log_lik', posterior_predictive='inv_rep')
print(az.loo(idata, pointwise=False))

rep = fit.stan_variable('inv_rep')
print('obs mean investment by round:', data['inv'].mean(0).round(2))
print('rep mean investment by round:', rep.mean((0, 1)).round(2))
