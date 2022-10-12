import numpy as np

import matplotlib.pyplot as plt
import arviz as az

np.random.seed(1)

ss = []   # 00
sb = []   # 01
bs = []   # 10
bb = []   # 11

for i in range(0, 100):
     a = []
     for j in range(0, 10):
         m_normala  = np.random.choice(a=[0, 1], p=[0.5, 0.5], size=1)[0]
         m_masluita = np.random.choice(a=[0, 1], p=[0.3, 0.7], size=1)[0]
         
         if m_normala==0 and m_masluita==0:
            a.append(1)
         if m_normala==0 and m_masluita==1:
            a.append(2)
         if m_normala==1 and m_masluita==0:
            a.append(3)
         if m_normala==1 and m_masluita==1:
            a.append(4)
      
     ss.append(a.count(1))
     sb.append(a.count(2))
     bs.append(a.count(3))
     bb.append(a.count(4))
     
az.plot_posterior({"ss": ss, "sb": sb, "bs" : bs, "bb" : bb})
plt.show()


