import numpy as np
import matplotlib.pyplot as plt

# (A)
n = 30
lambda0 = 2
sample = np.random.exponential(scale=1/lambda0, size=n)


# (B)
def log_likelihood(lmbda, data):
    return len(data) * np.log(lmbda) - lmbda * np.sum(data)

lambdas = np.linspace(0.5, 4, 200)
logL = [log_likelihood(l, sample) for l in lambdas]

lambda_hat = 1 / np.mean(sample)

plt.figure()
plt.subplot(121)
plt.plot(lambdas, logL, label="log-wiarygodność")
plt.axvline(lambda_hat, linestyle='--', label=f"λ̂ = {lambda_hat:.3f}")
plt.xlabel("λ")
plt.ylabel("ℓ(λ)")
plt.title("Log-wiarygodność")


# (C)
ns = [5, 30, 200]
lambdas = np.linspace(0.5, 4, 200)

plt.subplot(122)

for n in ns:
    data = np.random.exponential(scale=1/lambda0, size=n)
    logL = [log_likelihood(l, data)/n for l in lambdas]
    plt.plot(lambdas, logL, label=f"n={n}")
    plt.scatter(lambdas[np.argmax(logL)], max(logL))

plt.axvline(2, linestyle='--', label=f"λ0 = 2")
plt.xlabel("λ")
plt.ylabel("ℓ(λ)/n")
plt.title("Porównanie log-wiarygodności / n")


# (D)
ns = [5, 30, 200]
B = 1000

for n in ns:
    estimates = []
    
    for _ in range(B):
        data = np.random.exponential(scale=1/lambda0, size=n)
        lambda_hat = 1 / np.mean(data)
        estimates.append(lambda_hat)
    
    mean_estimate = np.mean(estimates)
    bias = mean_estimate - lambda0
    
    print(f"n={n}: średnia estymatora = {mean_estimate:.4f}, obciążenie = {bias:.4f}")

plt.legend()
plt.show()