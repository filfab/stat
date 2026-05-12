import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# (A)
rng = np.random.default_rng(1)
x = rng.normal(0, 1, 100)

# (B)
epsilons = [
    rng.normal(0, 0.5, 100),
    rng.normal(0, 0.1, 100),
    rng.normal(0, 1, 100),
    ]

# (C)

# # (D)
# plt.scatter(x, y)
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("Scatterplot of x and y")
# plt.show()

for eps, i in enumerate(epsilons):
    y = -1 + 0.5 * x + eps

    # (E)
    X = sm.add_constant(x)

    model = sm.OLS(y, X).fit()

    print(model.summary())

    # (F)
    plt.scatter(x, y)

    x_sorted = np.sort(x)
    y_hat = model.params[0] + model.params[1] * x_sorted
    plt.plot(x_sorted, y_hat, color="orange", label="Least Squares Line")

    y_true = -1 + 0.5 * x_sorted
    plt.plot(x_sorted, y_true, color="red", label="Population Line")

    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    # (G)
    X_poly = np.column_stack((x, x**2))
    X_poly = sm.add_constant(X_poly)

    poly_model = sm.OLS(y, X_poly).fit()

    print(poly_model.summary())