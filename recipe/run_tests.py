import os
from cmdstanpy import cmdstan_path, CmdStanModel

# specify locations of Stan program file and data
bernoulli_stan = os.path.join(cmdstan_path(), 'examples', 'bernoulli', 'bernoulli.stan')
bernoulli_data = os.path.join(cmdstan_path(), 'examples', 'bernoulli', 'bernoulli.data.json')

# instantiate a model; compiles the Stan program by default
bernoulli_model = CmdStanModel(stan_file=bernoulli_stan)

# obtain a posterior sample from the model conditioned on the data
bernoulli_fit = bernoulli_model.sample(chains=4, data=bernoulli_data)

# summarize the results (wraps CmdStan `bin/stansummary`):
bernoulli_fit.summary()
