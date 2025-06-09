import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange, CubicSpline
from numpy.polynomial import Polynomial
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Datos
pesos = np.array([8, 10, 12, 14, 16])
dosis = np.array([1.6, 2.0, 2.4, 2.8, 3.2])
peso_objetivo = 13

# 1. Interpolación de Newton (diferencias divididas)
def newton_interpolation(x, y, x0, x_eval=None):
    n = len(x)
    coef = np.copy(y)
    for j in range(1, n):
        coef[j:n] = (coef[j:n] - coef[j - 1]) / (x[j:n] - x[j - 1])
    def newton_poly(x_val):
        result = coef[0]
        prod = 1.0
        for i in range(1, n):
            prod *= (x_val - x[i - 1])
            result += coef[i] * prod
        return result
    if x_eval is not None:
        return newton_poly(x0), np.array([newton_poly(xi) for xi in x_eval])
    return newton_poly(x0)

# 2. Interpolación de Lagrange
def lagrange_interpolation(x, y, x0, x_eval=None):
    poly = lagrange(x, y)
    if x_eval is not None:
        return poly(x0), poly(x_eval)
    return poly(x0)

# 3. Spline cúbico
def spline_interpolation(x, y, x0, x_eval=None):
    spline = CubicSpline(x, y)
    if x_eval is not None:
        return spline(x0), spline(x_eval)
    return spline(x0)

# 4. Regresión polinomial (grado 2)
def regresion_polinomial(x, y, x0, grado=2, x_eval=None):
    x = x.reshape(-1, 1)
    poly_features = PolynomialFeatures(degree=grado)
    x_poly = poly_features.fit_transform(x)
    model = LinearRegression().fit(x_poly, y)
    x0_poly = poly_features.transform([[x0]])
    y0 = model.predict(x0_poly)[0]

    if x_eval is not None:
        x_eval_poly = poly_features.transform(x_eval.reshape(-1, 1))
        y_eval = model.predict(x_eval_poly)
        return y0, y_eval
    return y0

# Evaluación y resultados
x_plot = np.linspace(min(pesos), max(pesos), 200)

newton_result, newton_curve = newton_interpolation(pesos, dosis, peso_objetivo, x_plot)
lagrange_result, lagrange_curve = lagrange_interpolation(pesos, dosis, peso_objetivo, x_plot)
spline_result, spline_curve = spline_interpolation(pesos, dosis, peso_objetivo, x_plot)
regresion_result, regresion_curve = regresion_polinomial(pesos, dosis, peso_objetivo, 2, x_plot)

# Mostrar resultados
print(f"Dosis estimada para 13 kg:")
print(f"  - Interpolación de Newton:     {newton_result:.3f} ml")
print(f"  - Interpolación de Lagrange:   {lagrange_result:.3f} ml")
print(f"  - Interpolación Spline cúbico: {spline_result:.3f} ml")
print(f"  - Regresión polinomial (g=2):  {regresion_result:.3f} ml")

# Gráfica
plt.figure(figsize=(10, 6))
plt.plot(x_plot, newton_curve, '--', label='Newton')
plt.plot(x_plot, lagrange_curve, '-.', label='Lagrange')
plt.plot(x_plot, spline_curve, ':', label='Spline cúbico')
plt.plot(x_plot, regresion_curve, '-', label='Regresión (grado 2)')
plt.scatter(pesos, dosis, color='black', label='Datos reales')
plt.scatter([peso_objetivo], [newton_result], color='blue', label='Newton (13 kg)', zorder=5)
plt.scatter([peso_objetivo], [lagrange_result], color='orange', label='Lagrange (13 kg)', zorder=5)
plt.scatter([peso_objetivo], [spline_result], color='green', label='Spline (13 kg)', zorder=5)
plt.scatter([peso_objetivo], [regresion_result], color='red', label='Regresión (13 kg)', zorder=5)
plt.title("Interpolación de dosis de paracetamol (100 mg/ml)")
plt.xlabel("Peso del niño (kg)")
plt.ylabel("Dosis (ml)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
