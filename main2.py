import numpy as np
import matplotlib.pyplot as plt
import math

# Параметры
k = 0.45
c = 1.3
alpha = 0.003
R = 4
l = 2 * math.pi * R
uc = 0

def make_l(R):
    return 2 * math.pi * R

# Коэффициент при n^2 в экспоненте
def coef_n2(k, c, R):
    l = make_l(R)
    return k / c * (2 * math.pi / l) ** 2  # = k / (c * R^2)

# Константный член в экспоненте (теплообмен с окруж. средой)
def coef_const(alpha, c, R):
    return (2 * alpha) / (c * R)

def Fi(n, t, x, R, k, c, alpha):
    l = make_l(R)
    if n == 0:
        return 0.5 * math.exp(-coef_const(alpha, c, R) * t)
    Bn = -((-1)**n - 1) / (math.pi * n)
    omega_n = 2 * math.pi * n / l
    return Bn * np.sin(omega_n * x) * np.exp(-t * (coef_n2(k, c, R) * n ** 2 + coef_const(alpha, c, R)))

def partial_sum(x, t, N):
    result = Fi(0, t, x, R, k, c, alpha)
    for n in range(1, N + 1):
        result += Fi(n, t, x, R, k, c, alpha)
    return result

# График 1: зависимость U(x0, t)
def plot_temp_vs_time_multiple_x(x_list=[2.0, 5.0, 10.0], N=50):
    t_vals = np.linspace(0.01, 30, 300)
    plt.figure(figsize=(8, 5))

    for x0 in x_list:
        u_vals = [partial_sum(x0, t, N) for t in t_vals]
        plt.plot(t_vals, u_vals, label=f"x₀ = {x0:.1f}")

    plt.title("Температура w(x₀, t) при разных x₀")
    plt.xlabel("Время t")
    plt.ylabel("w(x₀, t)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# График 2: распределение U(x, t0)
def plot_temp_vs_x_multiple_t(t_list=[0.5, 5.0, 20.0], N=50):
    x_vals = np.linspace(0, l, 300)
    plt.figure(figsize=(8, 5))

    for t0 in t_list:
        u_vals = [partial_sum(x, t0, N) for x in x_vals]
        plt.plot(x_vals, u_vals, label=f"t = {t0}")

    plt.title("Распределение температуры w(x, t₀) при разных t₀")
    plt.xlabel("Координата x")
    plt.ylabel("w(x, t₀)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_temp_vs_time_multiple_x(x_list=[0,0.1,2.0, 5.0, 10.0], N=210)  # График 1
plot_temp_vs_x_multiple_t(t_list=[0,0.1,0.5, 5.0,10.0, 20.0], N=210)     # График 2

