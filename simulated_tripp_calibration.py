import numpy as np
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt

import src.model as models

data = pd.read_csv(
    "./data/simulated_data/prompt_fraction_0.20/partial_sample.csv"
)

covariance = np.load(
    "./data/simulated_data/prompt_fraction_0.20/covariance.npy"
)

observables_keys = ['mb', 'x1', 'c', 'z']
observables = data[observables_keys].to_numpy()

cfg = {
    "pars": [
        'Mb', 'alpha', 'beta', 'sig_int',
        'Om0', 'w0'
    ],
    "preset_values": {
        'H0': 70.,
        'wa': 0.
    },
    "init_values": {
        'Mb': -19.3,
        'alpha': -0.2,
        'beta': 3.1,
        'sig_int': 0.1,
        'Om0': 0.3,
        'w0': -1.
    }
}

init_values = np.array([
    cfg['init_values'][par] for par in cfg['pars']
])

np.random.seed(42)
init_values += 1e-1 * np.random.normal(size=len(init_values))

bounds = [(None,None), (None,None), (None,None), (0.0, 1.), (0.01, 0.99), (None,None)]
tripp_model = models.TrippModel(cfg)
result = opt.minimize(tripp_model, init_values, args=(observables, covariance), bounds=bounds)
values = result.x
uncertainties = np.sqrt(np.diag(result.hess_inv.todense()))
for res, sig, par in zip(values, uncertainties, cfg['pars']):
    print(f"{par} = {res:.3f} +/- {sig:.3f}")