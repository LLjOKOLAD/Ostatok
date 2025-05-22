import math
import numpy as np

# Параметры
k = 0.45
c = 1.3
alpha = 0.003
R = 4
l = 2 * math.pi * R


def make_l(R):
    return 2 * math.pi * R

# Коэффициент при n^2 в экспоненте
def coef_n2(k, c, R):
    l = make_l(R)
    return k / c * (2 * math.pi / l) ** 2  # = k / (c * R^2)

# Константный член в экспоненте (теплообмен с окруж. средой)
def coef_const(alpha, c, R):
    return (2 * alpha) / (c * R)

# Функция Фурье-компоненты с номером n
def Fi(n, t, x, R, k, c, alpha):
    l = make_l(R)
    if n == 0:
        return 0.5 * math.exp(-coef_const(alpha, c, R) * t)
    Bn = -((-1)**n - 1) / (math.pi * n**2)
    omega_n = 2 * math.pi * n / l
    return Bn * np.sin(omega_n * x) * np.exp(-t * (coef_n2(k, c, R) * n ** 2 + coef_const(alpha, c, R)))

# Частичная сумма ряда
def SUM(x, t, N, R, k, c, alpha):
    result = Fi(0, t, x, R, k, c, alpha)
    for n in range(1, N + 1):
        result += Fi(n, t, x, R, k, c, alpha)
    return result


def FI(n,t):
    exponent = t * ((2 * alpha) / (c * R) + (k * n ** 2) / (c * R ** 2))
    denominator = (n + 1) ** 2 * k * math.pi * t * math.exp(exponent)
    return (c * R ** 2) / denominator

# Подбор N по модулю одного члена
def num_of_iter(eps, t, x, R, k, c, alpha):
    n = 1
    while abs(FI(n,t)) >= eps:
        n += 1
    return n

# Уточнённый подбор N по разности сумм
def num_of_iter_exp(eps, t, x, R, k, c, alpha):
    Neps = num_of_iter(eps, t, x, R, k, c, alpha)
    Nexp = Neps
    while abs(SUM(x, t, Neps, R, k, c, alpha) - SUM(x, t, Nexp, R, k, c, alpha)) <= eps and Nexp > 0:
        Nexp -= 1
    return Nexp + 1

# Пример использования:
x = 4  # например
for t in [0.1, 5, 20]:
    print(f"t = {t}")
    for eps in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]:
        Neps = num_of_iter(eps, t, x, R, k, c, alpha)
        Nexp = num_of_iter_exp(eps, t, x, R, k, c, alpha)
        percent = (Nexp / Neps) * 100
        print(f"eps = {eps:.0e}, Neps = {Neps}, Nexp = {Nexp}, percent = {percent:.2f}%")
