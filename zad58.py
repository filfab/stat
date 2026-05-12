import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import statsmodels.formula.api as smf

import matplotlib.pyplot as plt

# =========================================================
# Wczytanie danych
# =========================================================

auto = pd.read_csv(
    "Auto.csv",
    na_values="?"
)

# usunięcie braków
auto = auto.dropna()

# =========================================================
# Konwersja kolumn numerycznych
# =========================================================

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

# =========================================================
# Kategorie
# =========================================================

auto["origin"] = auto["origin"].astype("category")

print(auto.dtypes)

train, valid = train_test_split(
    auto,
    test_size=0.2,
    random_state=1
)

# =========================================================
# MODEL 1
# year jako zmienna liczbowa
# =========================================================

model_numeric = smf.ols(
    formula="""
    mpg ~ horsepower + weight + acceleration + year
    """,
    data=train
).fit()

print("\n================ MODEL NUMERIC YEAR ================\n")
print(model_numeric.summary())

# =========================================================
# MODEL 2
# year jako zmienna kategoryczna
# =========================================================

model_category = smf.ols(
    formula="""
    mpg ~ horsepower + weight + acceleration + C(year)
    """,
    data=train
).fit()

print("\n================ MODEL CATEGORY YEAR ================\n")
print(model_category.summary())

# =========================================================
# Porównanie jakości modeli
# =========================================================

pred_numeric = model_numeric.predict(valid)
pred_category = model_category.predict(valid)

mse_numeric = mean_squared_error(
    valid["mpg"],
    pred_numeric
)

mse_category = mean_squared_error(
    valid["mpg"],
    pred_category
)

print("\n================ COMPARISON ================\n")

print("MSE - year numeric:", mse_numeric)
print("MSE - year category:", mse_category)

print("\nR^2 numeric:", model_numeric.rsquared)
print("R^2 category:", model_category.rsquared)

# =========================================================
# Wykres predykcji
# =========================================================

plt.figure(figsize=(10, 5))

# model numeric
plt.subplot(1, 2, 1)

plt.scatter(valid["mpg"], pred_numeric)

plt.plot(
    [valid["mpg"].min(), valid["mpg"].max()],
    [valid["mpg"].min(), valid["mpg"].max()]
)

plt.xlabel("True mpg")
plt.ylabel("Predicted mpg")

plt.title("Year Numeric")

# model category
plt.subplot(1, 2, 2)

plt.scatter(valid["mpg"], pred_category)

plt.plot(
    [valid["mpg"].min(), valid["mpg"].max()],
    [valid["mpg"].min(), valid["mpg"].max()]
)

plt.xlabel("True mpg")
plt.ylabel("Predicted mpg")

plt.title("Year Category")

plt.tight_layout()
plt.show()

# =========================================================
# Liczba poziomów zmiennej year
# =========================================================

print("\nLiczba różnych wartości year:")
print(auto["year"].nunique())

# =========================================================
# Wnioski
# =========================================================

print("\n================ WNIOSKI ================\n")

print("""
1. year jako zmienna liczbowa:
   - zakłada liniową zależność między year i mpg,
   - model jest prostszy,
   - ma mniej parametrów.

2. year jako zmienna kategoryczna:
   - każda wartość year dostaje osobny parametr,
   - model jest bardziej elastyczny,
   - może lepiej dopasować dane,
   - ale łatwiej o overfitting.

3. Jeśli liczba różnych wartości year byłaby bardzo duża:
   - model kategoryczny miałby ogromną liczbę parametrów,
   - interpretacja byłaby trudna,
   - mogłoby dojść do overfittingu.

4. Ogólna zasada:
   - zmienne porządkowe lub ciągłe zwykle traktujemy jako liczbowe,
   - zmienne bez naturalnej skali liczbowej traktujemy jako kategoryczne,
   - jeśli zależność nie jest liniowa i liczba poziomów jest mała,
     warto rozważyć kategorię.
""")