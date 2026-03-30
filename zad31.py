import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

N = 1000
mu = 260
sigma = 18
n = 9
N_plot = 100

def przedzial_ufnosci(mu, sigma, n, alpha=0.05):
    X = np.random.normal(loc=mu, scale=sigma, size=n)    
    S = np.std(X, ddof=1)
    X_bar = np.mean(X)
    
    t_alpha = stats.t.ppf(1 - alpha/2, df=n-1)    
    margin = t_alpha * S / np.sqrt(n)
    lower = X_bar - margin
    upper = X_bar + margin
    
    covers = (lower <= mu) and (mu <= upper)
    
    return (lower, upper), covers


# (A)
results = [przedzial_ufnosci(mu, sigma, n) for _ in range(N)]
intervals = [r[0] for r in results]
covers = [r[1] for r in results]

coverage = np.mean(covers)
widths = [upper - lower for lower, upper in intervals]
avg_width = np.mean(widths)
width_known_sigma = 2 * stats.norm.ppf(0.975) * sigma / np.sqrt(n)

print("Pokrycie:", coverage)
print("Średnia szerokość:", avg_width)
print("Szerokość przy znanym sigma:", width_known_sigma)


# (B)
results_plot = [przedzial_ufnosci(mu, sigma, n) for _ in range(N_plot)]

plt.figure()

red_count = 0
for i, ((lower, upper), cover) in enumerate(results_plot):
    color = 'blue' if cover else 'red'
    if not cover:
        red_count += 1
    plt.plot([lower, upper], [i, i], color=color)

plt.axvline(mu, color='black', linestyle='--')
plt.xlabel("Wartość")
plt.ylabel("Numer próby")
plt.title("Przedziały ufności 95%")
plt.show()

print("Procent czerwonych:", red_count / N_plot)