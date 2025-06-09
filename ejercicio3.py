import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange, CubicSpline
from sklearn.linear_model import LinearRegression

# Datos
profundidades = np.array([5, 10, 15, 20, 25])
humedades = np.array([32.5, 29.8, 27.1, 25.0, 23.2])
profundidad_objetivo = 18

# 1. Interpolación de Newton
def newton_interpolation(x, y, x0, x_eval=None):
    n = len(x)
    coef = np.copy(y)
    for j in range(1, n):
        coef[j:n] = (coef[j:n] - coef[j-1]) / (x[j:n] - x[j-1])
    def newton_poly(x_val):
        result = coef[0]
        prod = 1.0
        for i in range(1, n):
            prod *= (x_val - x[i-1])
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

# 4. Regresión lineal: f(x) = ax + b
def regresion_lineal(x, y, x0, x_eval=None):
    x = x.reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    a = model.coef_[0]
    b = model.intercept_
    y0 = model.predict([[x0]])[0]
    if x_eval is not None:
        y_eval = model.predict(x_eval.reshape(-1, 1))
        return y0, y_eval, a, b
    return y0, a, b

# Evaluaciones
x_plot = np.linspace(min(profundidades), max(profundidades), 200)

newton_result, newton_curve = newton_interpolation(profundidades, humedades, profundidad_objetivo, x_plot)
lagrange_result, lagrange_curve = lagrange_interpolation(profundidades, humedades, profundidad_objetivo, x_plot)
spline_result, spline_curve = spline_interpolation(profundidades, humedades, profundidad_objetivo, x_plot)
regresion_result, regresion_curve, a, b = regresion_lineal(profundidades, humedades, profundidad_objetivo, x_plot)

# Mostrar resultados
print(f"Humedad estimada a 18 cm de profundidad:")
print(f"  - Newton:     {newton_result:.3f} %")
print(f"  - Lagrange:   {lagrange_result:.3f} %")
print(f"  - Spline:     {spline_result:.3f} %")
print(f"  - Regresión lineal: {regresion_result:.3f} %")
print(f"    f(x) = {a:.4f}x + {b:.4f}")

# Gráfica
plt.figure(figsize=(10, 6))
plt.plot(x_plot, newton_curve, '--', label='Newton')
plt.plot(x_plot, lagrange_curve, '-.', label='Lagrange')
plt.plot(x_plot, spline_curve, ':', label='Spline cúbico')
plt.plot(x_plot, regresion_curve, '-', label=f'Regresión lineal (f(x) = {a:.2f}x + {b:.2f})')
plt.scatter(profundidades, humedades, color='black', label='Datos')
plt.scatter([profundidad_objetivo], [newton_result], color='blue', label='Newton (18 cm)', zorder=5)
plt.scatter([profundidad_objetivo], [lagrange_result], color='orange', label='Lagrange (18 cm)', zorder=5)
plt.scatter([profundidad_objetivo], [spline_result], color='green', label='Spline (18 cm)', zorder=5)
plt.scatter([profundidad_objetivo], [regresion_result], color='red', label='Regresión (18 cm)', zorder=5)
plt.title("Estimación de Humedad del Suelo")
plt.xlabel("Profundidad (cm)")
plt.ylabel("Humedad relativa (%)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
