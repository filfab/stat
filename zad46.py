import numpy as np
from numpy.random import Generator, MT19937
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

rng = Generator(MT19937())

n = 1000
m = 500
dims = [1, 2, 5, 10, 20, 50, 100]

accuracies = []

for d in dims:
    X_train = rng.normal(0, 1, size=(n, d))
    
    norms_train = np.linalg.norm(X_train, axis=1)
    r_d = np.median(norms_train)
    y_train = norms_train <= r_d
    
    X_test = np.random.randn(m, d)
    norms_test = np.linalg.norm(X_test, axis=1)
    y_test = norms_test <= r_d

    distances = cdist(X_test, X_train)
    nn_idx = np.argmin(distances, axis=1)
    y_pred = y_train[nn_idx]
    
    accuracies.append(np.mean(y_pred == y_test))


plt.figure()
plt.plot(dims, accuracies, marker='o')
plt.axhline(0.5, linestyle='--')
plt.xlabel("Wymiar d")
plt.ylabel("Dokładność")
plt.title("Degradacja KNN (K=1) w przestrzeniach wysokowymiarowych")
plt.grid(True)
plt.show()
