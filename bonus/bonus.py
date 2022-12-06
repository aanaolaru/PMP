import pymc3 as pm
import numpy as np

# valorile minime estimate pentru a se respecta conditiile din cerinta sunt:
#(1)
min_case = 2
min_statii = 3
#(2)
min_mese = 5

timp_total_comanda = 0
timp_total_preg = 0
timp_total_mananca = 0
nr1=0 
nr2=0

trafic_locatie = np.random.poisson(20, 100) 

for i in range(len(trafic_locatie)):

   model = pm.Model()

   with model:
     timp_comanda = pm.Normal('Tc', mu=1, sd = 0.5)
     timp_preg = pm.Exponential('Tp', lam = 1/2)
     timp_mananca = pm.Normal('Tm', mu=10, sd = 2)
     trace = pm.sample(trafic_locatie[i], chains=1)

   dictionary = {
              'timp_comanda': trace['Tc'].tolist(),
              'timp_preg': trace['Tp'].tolist(),
              'timp_mananca': trace['Tm'].tolist()
              }

   for j in range(len(dictionary['timp_comanda'])): 
      timp_total_comanda += dictionary['timp_comanda'][j]

   for j in range(len(dictionary['timp_preg'])):
      timp_total_preg += dictionary['timp_preg'][j]

   for j in range(len(dictionary['timp_mananca'])):
      timp_total_mananca += dictionary['timp_mananca'][j]

   timp_total_comanda /= min_case
   timp_total_preg /= min_statii
   timp_total_mananca /= min_mese
   timp_total_servire = timp_total_comanda + timp_total_preg

   if timp_total_servire <= 60: 
     nr1 +=1
   if timp_total_mananca <= 60:
     nr2 +=1


print(nr1/100) # first probability
print(nr2/100) # second probability
  
       


        



