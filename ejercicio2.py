import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange, CubicSpline
from sklearn.linear_model import LinearRegression

# Datos
velocidades = np.array([30, 50, 70, 90, 110])
consumos = np.array([11.2, 13.1, 14.8, 16.4, 19.0])
velocidad_objetivo = 80

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
x_plot = np.linspace(min(velocidades), max(velocidades), 200)

newton_result, newton_curve = newton_interpolation(velocidades, consumos, velocidad_objetivo, x_plot)
lagrange_result, lagrange_curve = lagrange_interpolation(velocidades, consumos, velocidad_objetivo, x_plot)
spline_result, spline_curve = spline_interpolation(velocidades, consumos, velocidad_objetivo, x_plot)
regresion_result, regresion_curve, a, b = regresion_lineal(velocidades, consumos, velocidad_objetivo, x_plot)

# Mostrar resultados
print(f"Consumo estimado a 80 km/h:")
print(f"  - Newton:     {newton_result:.3f} kWh/100km")
print(f"  - Lagrange:   {lagrange_result:.3f} kWh/100km")
print(f"  - Spline:     {spline_result:.3f} kWh/100km")
print(f"  - Regresión lineal: {regresion_result:.3f} kWh/100km")
print(f"    f(x) = {a:.4f}x + {b:.4f}")

# Gráfica
plt.figure(figsize=(10, 6))
plt.plot(x_plot, newton_curve, '--', label='Newton')
plt.plot(x_plot, lagrange_curve, '-.', label='Lagrange')
plt.plot(x_plot, spline_curve, ':', label='Spline cúbico')
plt.plot(x_plot, regresion_curve, '-', label=f'Regresión lineal (f(x) = {a:.2f}x + {b:.2f})')
plt.scatter(velocidades, consumos, color='black', label='Datos')
plt.scatter([velocidad_objetivo], [newton_result], color='blue', label='Newton (80 km/h)', zorder=5)
plt.scatter([velocidad_objetivo], [lagrange_result], color='orange', label='Lagrange (80 km/h)', zorder=5)
plt.scatter([velocidad_objetivo], [spline_result], color='green', label='Spline (80 km/h)', zorder=5)
plt.scatter([velocidad_objetivo], [regresion_result], color='red', label='Regresión (80 km/h)', zorder=5)
plt.title("Consumo energético estimado de un vehículo eléctrico")
plt.xlabel("Velocidad (km/h)")
plt.ylabel("Consumo (kWh/100 km)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
