import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az

model = pm.Model()

# 1)
with model:
    cutremur = pm.Bernoulli('C', 0.0005)
    incendiu_p = pm.Deterministic('I_p', pm.math.switch(cutremur, 0.03, 0.01))
    incendiu = pm.Bernoulli('I', p = incendiu_p)
    alarma_p = pm.Deterministic('A_p', pm.math.switch(cutremur, pm.math.switch(incendiu, 0.98, 0.02), pm.math.switch(incendiu, 0.95, 0.0001)))
    alarma = pm.Bernoulli('A', p = alarma_p)
    trace = pm.sample(20000,chains=1)

dictionary = {
              'cutremur': trace['C'].tolist(),
              'incendiu': trace['I'].tolist(),
              'alarma': trace['A'].tolist()
              }
df = pd.DataFrame(dictionary)

#2)
p_cutremur = df[((df['cutremur'] == 1) & (df['alarma'] == 1))].shape[0] / df[df['alarma'] == 1].shape[0]
print(p_cutremur)

#3)
p_incendiu = df[((df['incendiu'] == 1) & (df['alarma'] == 0))].shape[0] / df[df['alarma'] == 0].shape[0]
print(p_incendiu)

az.plot_posterior(trace)
plt.show()