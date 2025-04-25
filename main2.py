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

coef_n2 = k / (c * 16)
coef_const = (2 * alpha) / (c * R)

def Fi(n, t, x):
    if n == 0:
        return math.exp(-0.00115 * t)
    Bn = 2 / (math.pi * n)
    return Bn * np.sin(n * x / 4) * np.exp(-t * (coef_n2 * (n ** 2) + coef_const))

def partial_sum(x, t, N):
    result = Fi(0, t, x)  # нулевой член
    for n in range(2, N + 1, 2):
        result += Fi(n, t, x)
    return uc + result

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

