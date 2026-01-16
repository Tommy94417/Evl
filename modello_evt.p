import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genpareto

# 1. Simulazione dati di sinistri (Log-normal + Outliers)
np.random.seed(42)
data = np.random.lognormal(mean=2, sigma=1, size=1000)

# 2. Definizione della soglia (Threshold)
# In Solvency II, si sceglie spesso un quantile elevato (es. 95%)
threshold = np.percentile(data, 95)
excesses = data[data > threshold] - threshold

# 3. Fit della Generalized Pareto Distribution (GPD)
# c = shape parameter (xi), loc = 0, scale = sigma
shape, loc, scale = genpareto.fit(excesses)

print(f"Parametro di forma (xi): {shape:.4f}")
print(f"Parametro di scala (sigma): {scale:.4f}")

# 4. Calcolo del Value at Risk (VaR) al 99.5% (Target Solvency II)
prob_target = 0.995
n = len(data)
nu = len(excesses)
var_evt = threshold + (scale / shape) * (((n / nu) * (1 - prob_target))**(-shape) - 1)

print(f"VaR 99.5% stimato con EVT: {var_evt:.2f}")

# 5. Visualizzazione
plt.hist(excesses, bins=30, density=True, alpha=0.5, label='Eccessi osservati')
x = np.linspace(0, max(excesses), 100)
plt.plot(x, genpareto.pdf(x, shape, loc, scale), 'r-', label='GPD Fit')
plt.title("EVT: Fit della coda dei sinistri")
plt.legend()
plt.show()
