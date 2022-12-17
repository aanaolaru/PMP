import numpy as np
import arviz as az
import pymc3 as pm
import matplotlib.pyplot as plt

# ex 1
clusters = 3
n_cluster = [150, 200, 150]
n_total = sum(n_cluster)
means = [5, 0, 3]
std_devs = [2, 2, 2]
mix = np.random.normal(np.repeat(means, n_cluster),
np.repeat(std_devs, n_cluster))
az.plot_kde(np.array(mix))
np.savetxt( 'file.csv' , mix, delimiter=',' , fmt= '%.2f', comments= '', encoding= 'utf-8', )
plt.hist( mix, bins= 50, density= True, alpha= 0.5, color= 'b')
plt.show()


# ex 2
clusters = [2, 3, 4]
models = []
idatas = []
for cluster in clusters:
    # pt fiecare cluster
    with pm.Model() as model:
        p = pm.Dirichlet('p', a=np.ones(cluster))
        means = pm.Normal('means', mu=np.linspace(mix.min(), mix.max(), cluster),sd=10, shape=cluster, transform=pm.distributions.transforms.ordered)
        st_dev = pm.HalfNormal('st_dev', st_dev=10)
        y = pm.NormalMixture('y', w=p, mu=means, sd=st_dev)
        idata = pm.sample(1000, tune=2000, random_seed=123, return_inferencedata=True)
        idatas.append(idata)
        models.append(model)

# ex 3
model_waic = az.compare(dict(zip([str(c) for c in clusters], idatas)), method='BB-pseudo-BMA', ic="waic", scale="deviance")
model_loo = az.compare(dict(zip([str(c) for c in clusters], idatas)), method='BB-pseudo-BMA', ic="loo", scale="deviance")
print(model_waic)
print(model_loo)