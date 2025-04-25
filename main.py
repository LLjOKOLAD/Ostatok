import math
import numpy as np

# Параметры
k = 0.45
c = 1.3
alpha = 0.003
R = 4
l = 2 * math.pi * R


coef_n2 = k / (c * 16)
coef_const = (2 * alpha) / (c * R)

# Член ряда при четном n
def Fi(n, t, x):
    if n == 0:
        return math.exp(-0.00115 * t)
    Bn = 2 / (math.pi * n)
    return Bn * np.sin(n * x / 4) * np.exp(-t * (coef_n2 * (n ** 2) + coef_const))

# Частичная сумма ряда до N (включительно), по четным n
def SUM(x, t, N):
    result = Fi(0, t, x)  # нулевой член
    for n in range(2, N + 1, 2):
        result += Fi(n, t, x)
    return result

# Вычисление N, начиная с n=2, при котором последний член меньше eps
def num_of_iter(eps, t, x):
    n = 2
    while abs(Fi(n, t, x)) > eps:
        n += 2
    return n

# Экспериментально уточнённое N
def num_of_iter_exp(eps, t, x):
    Neps = num_of_iter(eps, t, x)
    Nexp = Neps
    while abs(SUM(x, t, Neps) - SUM(x, t, Nexp)) <= eps and Nexp > 0:
        Nexp -= 2
    return Nexp + 2

# Пример использования:
x = 4  # например
for t in [0.1, 5, 20]:
    print(f"t = {t}")
    for eps in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]:
        Neps = num_of_iter(eps, t, x)
        Nexp = num_of_iter_exp(eps, t, x)
        print(f"eps = {eps:.0e}, Neps = {Neps}, Nexp = {Nexp}")
