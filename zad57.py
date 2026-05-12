import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import statsmodels.api as sm
import statsmodels.formula.api as smf

auto = pd.read_csv("Auto.csv", na_values="?")

auto = auto.dropna()

numeric_cols = [
    "mpg",
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "year"
]

for col in numeric_cols:
    auto[col] = pd.to_numeric(auto[col])

auto["origin"] = auto["origin"].astype("category")

print(auto.head())
print(auto.info())

# =========================================================
# Podział train / validation
# =========================================================

train, valid = train_test_split(
    auto,
    test_size=0.2,
    random_state=1
)

# =========================================================
# (a) Regresja liniowa mpg ~ horsepower
# =========================================================

model1 = smf.ols(
    formula="mpg ~ horsepower",
    data=train
).fit()

print("\n================ MODEL 1 ================\n")
print(model1.summary())

# =========================================================
# Wykres scatter + regresja
# =========================================================

sns.scatterplot(data=train, x="horsepower", y="mpg")

x_vals = np.linspace(
    train["horsepower"].min(),
    train["horsepower"].max(),
    100
)

y_vals = (
    model1.params["Intercept"]
    + model1.params["horsepower"] * x_vals
)

plt.plot(x_vals, y_vals)

plt.title("mpg vs horsepower")

plt.savefig(f"57-1.png")

# =========================================================
# Diagnostyka reszt
# =========================================================

fig = sm.graphics.plot_regress_exog(model1, "horsepower")

plt.savefig(f"57-2.png")

sm.qqplot(model1.resid, line="45")
plt.title("QQ Plot")

plt.savefig(f"57-3.png")

# =========================================================
# (b) Różne modele
# =========================================================

# model wieloliniowy
model2 = smf.ols(
    formula="""
    mpg ~ cylinders + displacement + horsepower
          + weight + acceleration + year + origin
    """,
    data=train
).fit()

# model z interakcją
model3 = smf.ols(
    formula="""
    mpg ~ horsepower * weight + year + origin
    """,
    data=train
).fit()

# model z logarytmem
model4 = smf.ols(
    formula="""
    mpg ~ I(np.log(horsepower))
          + weight + year + origin
    """,
    data=train
).fit()

# model z pierwiastkiem
model5 = smf.ols(
    formula="""
    mpg ~ I(np.sqrt(weight))
          + horsepower + year + origin
    """,
    data=train
).fit()

# model wielomianowy
model6 = smf.ols(
    formula="""
    mpg ~ horsepower
          + I(horsepower**2)
          + weight
          + year
          + origin
    """,
    data=train
).fit()

# =========================================================
# Summary modeli
# =========================================================

print("\n================ MODEL 2 ================\n")
print(model2.summary())

print("\n================ MODEL 3 ================\n")
print(model3.summary())

print("\n================ MODEL 4 ================\n")
print(model4.summary())

print("\n================ MODEL 5 ================\n")
print(model5.summary())

print("\n================ MODEL 6 ================\n")
print(model6.summary())

# =========================================================
# Residual plots dla najlepszego modelu
# =========================================================

fig = sm.graphics.plot_regress_exog(
    model6,
    "horsepower"
)


plt.savefig(f"57-3.png")

sm.qqplot(model6.resid, line="45")
plt.title("QQ Plot - Model 6")

plt.savefig(f"57-4.png")

# =========================================================
# (c) Błąd generalizacji
# =========================================================

models = {
    "model1": model1,
    "model2": model2,
    "model3": model3,
    "model4": model4,
    "model5": model5,
    "model6": model6
}

print("\n================ VALIDATION MSE ================\n")

best_model = None
best_mse = np.inf

for name, model in models.items():

    pred = model.predict(valid)

    mse = mean_squared_error(
        valid["mpg"],
        pred
    )

    print(f"{name}: {mse:.4f}")

    if mse < best_mse:
        best_mse = mse
        best_model = name

print("\nNajlepszy model:", best_model)
print("Najmniejszy MSE:", best_mse)

# =========================================================
# Predykcje najlepszego modelu
# =========================================================

best_predictions = models[best_model].predict(valid)

plt.scatter(valid["mpg"], best_predictions)

plt.xlabel("True mpg")
plt.ylabel("Predicted mpg")

plt.title(f"Predictions - {best_model}")

plt.plot(
    [valid["mpg"].min(), valid["mpg"].max()],
    [valid["mpg"].min(), valid["mpg"].max()]
)


plt.savefig(f"57-5.png")