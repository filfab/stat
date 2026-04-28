import numpy as np
from numpy.random import Generator, MT19937
import matplotlib.pyplot as plt
from scipy.special import gamma

rng = Generator(MT19937())

dims = range(1, 21)
N = 100_000

V_exact = []
V_mc = []
ratios = []

for d in dims:
    Vd = (np.pi ** (d / 2)) / gamma(d / 2 + 1)
    V_exact.append(Vd)

    # --- Monte Carlo ---
    X = rng.uniform(-1, 1, size=(N, d))
    norms = np.linalg.norm(X, axis=1)

    inside = np.sum(norms <= 1)
    Vd_mc = (inside / N) * (2 ** d)
    V_mc.append(Vd_mc)

    ratios.append(Vd / (2 ** d))

plt.figure()
plt.subplot(1, 2, 1)
plt.plot(dims, V_exact, marker='o', label="Analityczna")
plt.plot(dims, V_mc, marker='x', linestyle='--', label="Monte Carlo")
plt.xlabel("Wymiar d")
plt.ylabel("Objętość hiperkuli")
plt.title("Objętość hiperkuli w funkcji wymiaru")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(dims, ratios, marker='o')
plt.xlabel("Wymiar d")
plt.ylabel("V_d / 2^d")
plt.title("Udział hiperkuli w hiperkostce")
plt.grid(True)
plt.show()
