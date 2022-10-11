import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

np.random.seed(1)

a=[]
x = stats.expon.rvs(0, 1/4, 1)    # latenta 1/4 pe care o adaug la gamma
z = stats.uniform.rvs(0, 1, size=10000) # Distributie uniforma intre 0 si 1, 1000 samples
g1 = stats.gamma.rvs(4, 0, 1/3,size=10000) 
g2 = stats.gamma.rvs(4, 0, 1/2,size=10000)
g3 = stats.gamma.rvs(5, 0, 1/2,size=10000)
g4 = stats.gamma.rvs(5, 0, 1/3,size=10000)

for i in range(10000):
   if z[i] < 0.25: # server 1
      a.append(g1[i] + x)
      a.append(g2[i] + x)
   if z[i] < 0.3:
      a.append(g3[i] + x)   

mean = np.mean(a)
print(mean)
stdev=np.std(a)
print(stdev)

az.plot_posterior({'a':a}) # Afisarea aproximarii densitatii probabilitatilor, mediei, intervalului etc. variabilelor x,y,z
plt.show() 