import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ========== МАТЕМАТИЧЕСКАЯ ЧАСТЬ (оптимизировано для памяти) ==========

def explicit_step(omega, c, kappa, alpha, r0, hx, ht):
    """Один временной шаг явной схемы (без аллокации большого массива)."""
    I = len(omega) - 1
    omega_new = np.empty_like(omega)
    coeff = ht / c
    reaction = 2.0 * alpha / r0
    hx2 = hx * hx

    # внутренние узлы i = 1..I-1
    for i in range(1, I):
        lapl = (omega[i + 1] - 2.0 * omega[i] + omega[i - 1]) / hx2
        omega_new[i] = omega[i] + coeff * (kappa * lapl - reaction * omega[i])

    # граничное условие на новом уровне
    omega_new[0] = 0.5 * (omega_new[1] + omega_new[I - 1])
    omega_new[I] = omega_new[0]
    return omega_new


def solve_explicit_sampled_time(c, kappa, alpha, r0, L, T, Nx, Nt, psi_func, x_points, progress=None):
    """
    Решаем схему и возвращаем временные ряды omega(t) в узлах, соответствующих x_points.
    Память: O(Nt * m) где m = len(x_points) (а не O(Nt * Nx)).
    Вход: x_points в тех же единицах, что и L, либо дроби [0,1] (будут интерпретированы как доли L).
    """
    # сетка
    hx = L / Nx
    ht = T / Nt
    x_grid = np.linspace(0.0, L, Nx + 1)

    # перевод x_points -> индексы i0
    idxs = []
    for x0 in x_points:
        # если ввод в [0,1] воспринимать как долю длины
        if 0.0 <= x0 <= 1.0:
            x_abs = x0 * L
        else:
            x_abs = x0
        i0 = int(round(x_abs / L * Nx))
        i0 = min(max(i0, 0), Nx)  # clamp
        idxs.append(i0)

    m = len(idxs)
    # храним m временных рядов
    results = np.zeros((Nt + 1, m))

    # начальное условие
    omega = psi_func(x_grid)
    omega[-1] = omega[0]

    # записать t=0
    for j, i0 in enumerate(idxs):
        results[0, j] = omega[i0]

    # основной цикл, сохраняем только нужные индексы
    for k in range(1, Nt + 1):
        omega = explicit_step(omega, c, kappa, alpha, r0, hx, ht)
        for j, i0 in enumerate(idxs):
            results[k, j] = omega[i0]

        # обновление прогресса редко, чтобы не тормозить
        if progress and (k % max(1, Nt // 100) == 0 or k == Nt):
            progress['value'] = k / Nt * 100
            progress.update()

    if progress:
        progress['value'] = 0
    t_vals = np.linspace(0.0, T, Nt + 1)
    return t_vals, results


def solve_explicit_sampled_space(c, kappa, alpha, r0, L, T, Nx, Nt, psi_func, t_points, progress=None):
    """
    Решаем схему и возвращаем пространственные профили omega(x) в моменты t_points.
    Память: O(Nx * m) где m = len(t_points).
    """
    hx = L / Nx
    ht = T / Nt
    x_grid = np.linspace(0.0, L, Nx + 1)

    # перевод t_points -> индексы k0
    k_idxs = []
    for t0 in t_points:
        if t0 < 0:
            t0 = 0.0
        if t0 > T:
            t0 = T
        k0 = int(round(t0 / T * Nt))
        k0 = min(max(k0, 0), Nt)
        k_idxs.append(k0)

    m = len(k_idxs)
    results = np.zeros((m, Nx + 1))  # m профилей

    # начальное условие
    omega = psi_func(x_grid)
    omega[-1] = omega[0]

    # если нужно t=0, записываем
    for j, k0 in enumerate(k_idxs):
        if k0 == 0:
            results[j, :] = omega.copy()

    # основной цикл, сохраняем только при совпадении временного шага
    for k in range(1, Nt + 1):
        omega = explicit_step(omega, c, kappa, alpha, r0, hx, ht)
        for j, k0 in enumerate(k_idxs):
            if k == k0:
                results[j, :] = omega.copy()

        if progress and (k % max(1, Nt // 100) == 0 or k == Nt):
            progress['value'] = k / Nt * 100
            progress.update()

    if progress:
        progress['value'] = 0
    x_vals = x_grid
    return x_vals, results


# ========== Визуализация (адаптированы для новых solve_...) ==========

def plot_temp_vs_time(x_points, R, kappa, c, alpha, r0, Nx, Nt, T, ax, progress=None):
    L = 2.0 * np.pi * R
    # функция начального условия (можно заменить)
    psi_func = lambda x: np.sin(2.0 * np.pi * x / L)
    t_vals, results = solve_explicit_sampled_time(c, kappa, alpha, r0, L, T, Nx, Nt, psi_func, x_points, progress)
    ax.clear()
    for j, x0 in enumerate(x_points):
        ax.plot(t_vals, results[:, j], label=f"x₀ = {x0}")
    ax.set_title("ω(x₀, t)")
    ax.set_xlabel("t")
    ax.set_ylabel("ω")
    ax.legend()
    ax.grid(True)


def plot_temp_vs_x(t_points, R, kappa, c, alpha, r0, Nx, Nt, T, ax, progress=None):
    L = 2.0 * np.pi * R
    psi_func = lambda x: np.sin(2.0 * np.pi * x / L)
    x_vals, results = solve_explicit_sampled_space(c, kappa, alpha, r0, L, T, Nx, Nt, psi_func, t_points, progress)
    ax.clear()
    for j, t0 in enumerate(t_points):
        ax.plot(x_vals, results[j, :], label=f"t₀ = {t0}")
    ax.set_title("ω(x, t₀)")
    ax.set_xlabel("x")
    ax.set_ylabel("ω")
    ax.legend()
    ax.grid(True)


# ========== Интерфейс (остальное оставляем как у тебя, только вызываем новые функции) ==========

# (здесь вставьте UI-код — он идентичен твоему предыдущему; в обработчиках вызов заменён:
#   plot_temp_vs_time(..., R, kappa, c, alpha, r0, Nx, Nt, T, ax, progress_bar)
#   plot_temp_vs_x(..., R, kappa, c, alpha, r0, Nx, Nt, T, ax, progress_bar)
# )

# Ниже — пример применения функций автономно (для отладки):
if __name__ == "__main__":
    # тестовый запуск маленькой сетки
    R = 4.0
    alpha = 0.003
    kappa = 0.45
    c = 1.3
    r0 = 4.0
    Nx = 800       # можно и большое, память под профиль O(Nx * m)
    Nt = 2000      # можно и большое, время работы увеличится
    T = 0.5

    # проверка: x_points вводим как абсолютные координаты в [0, L]
    L = 2.0 * np.pi * R
    x_points = [0.0, 0.2 * L, 0.5 * L, 0.8 * L]
    import time
    t0 = time.time()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    plot_temp_vs_time(x_points, R, kappa, c, alpha, r0, Nx, Nt, T, ax, progress=None)
    plt.show()
    print("time:", time.time() - t0)
