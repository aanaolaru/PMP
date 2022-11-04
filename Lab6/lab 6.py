import arviz as az
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pymc3 as pm

# 1
data = pd.read_csv(r'C:\Users\Alina\Desktop\PMP\Lab6\data.csv')
data.dropna(inplace=True)
educ_list = data['educ_cat'].tolist()
momage_list = data['momage'].tolist()
plt.scatter(educ_list, momage_list, c="blue")
plt.show()


#2
def phi(x) :
	s = 0.1 
	return np.append(1, np.exp(-(x - np.arange(0, 1 + s, s)) ** 2 / (2 * s * s)))


basic_model = pm.Model()

with basic_model:

  PHI = np.array([phi(e) for e in educ_list])
  print ("PHI = ", PHI)
  
  #alpha = pm.Normal('alpha', mu=0, sd=10)
  #beta = pm.Normal('beta', mu=0, sd=1)
  #Sigma_N = pm.HalfCauchy('sigma', 5)

  alpha = 0.1 
  beta = 9.0  
  Sigma_N = np.linalg.inv(alpha * np.identity(PHI.shape[1]) + beta * np.dot(PHI.T, PHI))
  mu_N = beta * np.dot(Sigma_N, np.dot(PHI.T, momage_list))
  print ("mu_N = ", mu_N)


xlist = np.arange(0, 1, 0.01)
bayesian_lr = [np.dot(mu_N, phi(x)) for x in xlist]

plt.plot(xlist, bayesian_lr, 'b')
plt.plot(educ_list, momage_list, 'o')
plt.show()

