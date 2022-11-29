import math
import numpy as np
import pandas as pd
import pymc3 as pm
import matplotlib.pyplot as plt
import arviz as az

if __name__ == "__main__":

    data = pd.read_csv('./Admission.csv')

    gre = data['GRE'].values
    gpa = data['GPA'].values
    admission = data['Admission'].values
    y_0 = pd.Categorical(admission).codes

    x_n = ['GRE', 'GPA']
    first_axis_x1 = data[x_n].values
    
    #1
    model=pm.Model()
    with model:
          alpha = pm.Normal('alpha', mu=0, sd=10)
          beta0 = pm.Normal('beta0', mu = 0, sd = 10)
          beta1 = pm.Normal('beta1', mu=0, sd=10)
          beta2 = pm.Normal('beta2', mu=0, sd=10)
    
          mu = beta0 + pm.math.dot(beta1, gre) + pm.math.dot(beta2, gpa)
          theta = pm.Deterministic('theta', 1 / (1 + pm.math.exp(-mu)))
          bd = pm.Deterministic('bd', -alpha/beta1 - beta0/beta1 * first_axis_x1[:,0])
          p = pm.Bernoulli('p', p=theta, observed=y_0)
          trace = pm.sample(2000, tune=2000, cores=4)
          trace = pm.sample(return_inferencedata=True)

    #2
    posterior = trace.posterior.stack(samples=("chain", "draw"))
    theta = posterior['mu'].mean("samples")
    idx = np.argsort(gre)
    plt.plot(gre[idx], theta[idx], color='C2', lw=3)
    plt.vlines(posterior['bd1'].mean(), 0, 1, color='k')
    bd_hpd = az.hdi(posterior['bd1'].values)
    plt.fill_betweenx([0, 1], bd_hpd[0], bd_hpd[1], color='k', alpha=0.5)
    plt.scatter(gre, np.random.normal(admission, 0.02), marker='.', color=[f'C{x}' for x in admission])
    az.plot_hdi(gre, posterior['mu'].T, color='C2', smooth=False)
    locs, _ = plt.xticks()
    plt.xticks(locs, np.round(locs + gpa.mean(), 1))
    plt.xlabel("GRE")
    plt.ylabel('mu', rotation=0)
    plt.show()

    posterior = trace.posterior.stack(samples=("chain", "draw"))
    theta = posterior['mu'].mean("samples")
    idx = np.argsort(gpa)
    plt.plot(gpa[idx], theta[idx], color='C2', lw=3)
    plt.vlines(posterior['bd2'].mean(), 0, 1, color='k')
    bd_hpd = az.hdi(posterior['bd1'].values)
    plt.fill_betweenx([0, 1], bd_hpd[0], bd_hpd[1], color='k', alpha=0.5)
    plt.scatter(gpa, np.random.normal(admission, 0.02), marker='.', color=[f'C{x}' for x in admission])
    az.plot_hdi(gpa, posterior['mu'].T, color='C2', smooth=False)
    locs, _ = plt.xticks()
    plt.xticks(locs, np.round(locs + gpa.mean(), 1))
    plt.xlabel("GPA")
    plt.ylabel('mu', rotation=0)
    plt.show() 
