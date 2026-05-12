import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

rng = np.random.default_rng(10)

x1 = rng.uniform(0, 1, size=100)

x2 = 0.5 * x1 + rng.normal(size=100) / 10

y = 2 + 2 * x1 + 0.3 * x2 + rng.normal(size=100)

# (B)
print("korelacja x1 x2: ", np.corrcoef(x1, x2))

plt.scatter(x1, x2)
plt.xlabel("x1")
plt.ylabel("x2")

plt.savefig("56.png")

# (C)
X = np.column_stack((x1, x2))
X = sm.add_constant(X)

model = sm.OLS(y, X).fit()

print(model.summary(),end="\n\n\n")

# (D)
X1 = sm.add_constant(x1)

model1 = sm.OLS(y, X1).fit()

print(model1.summary(),end="\n\n\n")

# (E)
X2 = sm.add_constant(x2)

model2 = sm.OLS(y, X2).fit()

print(model2.summary(),end="\n\n\n")

# (G)
x1 = np.concatenate([x1, [0.1]])
x2 = np.concatenate([x2, [0.8]])
y = np.concatenate([y, [6]])

# (C)
X = np.column_stack((x1, x2))
X = sm.add_constant(X)

model = sm.OLS(y, X).fit()

print(model.summary(),end="\n\n\n")

# (D)
X1 = sm.add_constant(x1)

model1 = sm.OLS(y, X1).fit()

print(model1.summary(),end="\n\n\n")

# (E)
X2 = sm.add_constant(x2)

model2 = sm.OLS(y, X2).fit()

print(model2.summary(),end="\n\n\n")
