import numpy as np
from numpy.random import Generator, MT19937
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

rng = Generator(MT19937())

n = 200
x = np.random.rand(n, 1)
epsilon = np.random.normal(0, 0.3, size=(n, 1))
y = np.sin(2 * np.pi * x) + epsilon

degrees = list(range(1, 16))
mse_train = []
mse_cv = []

kf = KFold(n_splits=5, shuffle=True, random_state=0)

for d in degrees:
    poly = PolynomialFeatures(degree=d)
    X_poly = poly.fit_transform(x)
    
    # --- trening ---
    model = LinearRegression()
    model.fit(X_poly, y)
    y_pred_train = model.predict(X_poly)
    mse_train.append(mean_squared_error(y, y_pred_train))
    
    # --- 5-fold CV ---
    cv_errors = []
    
    for train_idx, val_idx in kf.split(X_poly):
        X_tr, X_val = X_poly[train_idx], X_poly[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        model_cv = LinearRegression()
        model_cv.fit(X_tr, y_tr)
        y_pred_val = model_cv.predict(X_val)
        
        cv_errors.append(mean_squared_error(y_val, y_pred_val))
    
    mse_cv.append(np.mean(cv_errors))

# --- wykres ---
plt.figure()
plt.plot(degrees, mse_train, marker='o', label='MSE treningowe')
plt.plot(degrees, mse_cv, marker='x', label='MSE CV (5-fold)')
optimum = degrees[mse_cv.index(min(mse_cv))]
plt.axvline(optimum, color="red", linestyle="--", label=f'OPTIMAL: d={optimum}')
plt.xlabel("Stopień wielomianu d")
plt.ylabel("MSE")
plt.title("Bias-Variance tradeoff (regresja wielomianowa)")
plt.legend()
plt.grid(True)
plt.show()

# --- najlepszy stopień ---
best_d = degrees[np.argmin(mse_cv)]
print(f"Optymalny stopień (wg CV): d = {best_d}")