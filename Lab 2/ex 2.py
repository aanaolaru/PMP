import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

np.random.seed(1)

a=[]
x = stats.expon.rvs(0, 1/4, 1)    # latenta 1/4 pe care o adaug la gamma
z = stats.uniform.rvs(0, 1, size=10000) # Distributie uniforma intre 0 si 1, 1000 samples
g1 = stats.gamma.rvs(4, 0, 1/3) 
g2 = stats.gamma.rvs(4, 0, 1/2)
g3 = stats.gamma.rvs(5, 0, 1/2)
g4 = stats.gamma.rvs(5, 0, 1/3)

