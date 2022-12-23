import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import scipy.stats as stats
from  statistics import *


# ex 1
def posterior_grid(grid_points=50, heads=6, tails=9):

    grid = np.linspace(0, 1, grid_points)
    prior = (grid <= 0.5).astype(int)
    likelihood = stats.binom.pmf(heads, heads+tails, grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return grid, posterior


data = np.repeat([0, 1], (15, 7)) 
points = 20
h = data.sum()
t = len(data) - h
grid, posterior = posterior_grid(points, h, t)
plt.plot(grid, posterior, 'o-')
plt.title(f'heads = {h}, tails = {t}')
plt.yticks([])
plt.xlabel('θ')
plt.show()


# ex 2

def pi_estim(N):

  erros_list = []
  error_mean = 0
  std_deviation = 0

  for i in range(2):
     x, y = np.random.uniform(-1, 1, size=(2, N))
     inside = (x**2 + y**2) <= 1
     pi = inside.sum()*4/N
     error = abs((pi - np.pi) / pi) * 100
     outside = np.invert(inside)
     plt.figure(figsize=(8, 8))
     plt.plot(x[inside], y[inside], 'b.')
     plt.plot(x[outside], y[outside], 'r.')
     plt.plot(0, 0, label=f'π*= {pi:4.3f}\n error = {error:4.3f}', alpha=0)
     plt.axis('square')
     plt.xticks([])
     plt.yticks([])
     plt.legend(loc=1, frameon=True, framealpha=0.9)

     error_mean += error   # suma erori
     erros_list.append(error)

  error_mean = error_mean / 2
  std_deviation = stdev(erros_list)
  plt.errorbar(x,y,error_mean)
  plt.show()
  return error_mean, erros_list, std_deviation
  

N = [100, 1000, 10000]

error_mean, erros_list, std_deviation = pi_estim(N[0])
print("Eroarea medie pentru N = 100: ", error_mean)
print("Deviatia standard pentru N = 100: ", (std_deviation))



error_mean, erros_list, std_deviation = pi_estim(N[1])
print("Eroarea medie pentru N = 1000: ", error_mean)
print("Deviatia standard pentru N = 1000: ", (std_deviation))


error_mean, erros_list, std_deviation = pi_estim(N[2])
print("Eroarea medie pentru N = 10000: ", error_mean)
print("Deviatia standard pentru N = 10000: ", (std_deviation))


#ex 3
def metropolis(func, draws=10000):
    trace = np.zeros(draws)
    old_x = 0.5 # func.mean()
    old_prob = func.pdf(old_x)
    delta = np.random.normal(0, 0.5, draws)

    for i in range(draws):
        new_x = old_x + delta[i]
        new_prob = func.pdf(new_x)
        acceptance = new_prob / old_prob
        if acceptance >= np.random.random():
            trace[i] = new_x
            old_x = new_x
            old_prob = new_prob
        else:
            trace[i] = old_x
    return trace

n_param = [(1, 3), (2,4)]

for i, j in n_param:
    func = stats.beta(i, j)
    trace = metropolis(func=func)
    x = np.linspace(0.01, .99, 100)
    y = func.pdf(x)
    plt.xlim(0, 1)
    plt.plot(x, y, 'C1-', lw=3, label='Distributia adevarata')
    plt.hist(trace[trace > 0], bins=25, density=True, label='Distributia estimata')
    plt.xlabel('x')
    plt.ylabel('pdf(x)')
    plt.yticks([])
    plt.legend()