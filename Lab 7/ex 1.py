import math
import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az
import matplotlib.pyplot as plt

data = pd.read_csv('Prices.csv')

price = data['Price'].values
speed = data['Speed'].values
hardDrive = data['HardDrive'].values
ram = data['Ram'].values
premium = data['Premium'].values

#fig, axes = plt.subplots(2, 2, sharex=False, figsize=(10, 8))
#axes[0,0].scatter(speed, price, alpha=0.6)
#axes[0,1].scatter(hardDrive, price, alpha=0.6)
#axes[1,0].scatter(ram, price, alpha=0.6)
#axes[1,1].scatter(premium, price, alpha=0.6)
#axes[0,0].set_ylabel("Price")
#axes[0,0].set_xlabel("Speed")
#axes[0,1].set_xlabel("HardDrive")
#axes[1,0].set_xlabel("Ram")
#axes[1,1].set_xlabel("Premium")
#plt.savefig('price_correlations.png')

# 1
with pm.Model() as model:
    alfa2 = pm.Normal('alfa_tmp', mu=0, sd=10)
    beta1 = pm.Normal('beta1', mu=0, sd=1)
    beta2 = pm.Normal('beta2', mu=0, sd=1)
    epsilon = pm.HalfCauchy('eps', 5)
    
    mu2 = alfa2 + pm.math.dot(speed, beta1) + pm.math.dot(math.log(hardDrive), beta2)
    alfa = pm.Deterministic('alph', alfa2 - pm.math.dot(np.mean(speed), beta1) + pm.math.dot(np.mean(math.log(hardDrive)), beta2))
    y_pred = pm.Normal('y_pred', mu=mu2, sd=epsilon, observed=price)
    idata_mlr = pm.sample(2000, return_inferencedata=True, chains=2)

az.plot_trace(idata_mlr, var_names=['alfa', 'beta1', 'beta2', 'epsilon'])


# 2
az.plot_posterior(
        {"alpha": idata_mlr['alpha'], "beta1": idata_mlr['beta1'], "beta2": idata_mlr['beta2'], "epsilon": idata_mlr['epsilon']},
        hdi_prob=0.95,)
plt.show()


# 3
# Daca analizam rezultatele obtinute pentru beta1 si beta2, putem spune ca frecventa procesorului
# si marimea hard disk-ului au o influenta semnificativa asupra pretului de vanzare 

#4
with pm.Model as model:
     alpha = pm.Normal("alpha", mu=np.mean(price), sd=2)
     beta1 = pm.Normal("beta1", mu=-5, sd=10)
     beta2 = pm.Normal("beta2", mu=8, sd=12)
     epsilon = pm.HalfCauchy("epsilon", np.std(price))
     mu2 = pm.Deterministic("mu2", alpha + beta1 * [33 for i in price] + beta2 * [np.log(540) for i in price])

     y_pred = pm.Normal("y_pred", mu=mu2, sd=epsilon, observed=price)
     idata_g = pm.sample(5000, tune=5000, return_inferencedata=True)

az.plot_trace(idata_g, var_names=["alpha", "beta1", "beta2", "sigma"])
plt.show()