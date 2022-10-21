import matplotlib.pyplot as plt
import pymc3 as pm
import pandas as pd
import arviz as az

model = pm.Model()
alpha = 10

with model:
    clienti = pm.Poisson('C', mu=20)
    timp_comanda = pm.Normal('Tc', mu=1, sd=0.5)
    timp_preg = pm.Exponential('Tp', lam=1/alpha)
    trace = pm.sample(20000, chains=1)

az.plot_posterior(trace)
plt.show()