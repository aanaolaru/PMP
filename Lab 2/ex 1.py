import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

np.random.seed(1)


a=[]
x = stats.expon.rvs(0, 1/4, 10000)     #  random time
y = stats.expon.rvs(0, 1/6, 10000)     #  random time
b=x+y
z=stats.binom.rvs(1, 0.4, size=10000) # = 1 cu 40% sanse si 0 cu 60% sanse 

for i in range(10000):
    if z[i]==1: #sub 40
        a.append(x[i])
    if z[i]==0:
        a.append(y[i])  
    

print(x)
meann = np.mean(a)
print(meann)
stdd=np.std(a)
print(stdd)

az.plot_posterior({'a':a}) # Afisarea aproximarii densitatii probabilitatilor, mediei, intervalului etc. variabilelor x,y,z
plt.show() 