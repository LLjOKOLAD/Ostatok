import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import math

# =================== МАТЕМАТИЧЕСКАЯ ЧАСТЬ ===================

def explicit_step(omega, c, kappa, alpha, r0, hx, ht):
    """Один временной шаг явной схемы для вектора omega (узлы 0..Nx)."""
    I = len(omega) - 1
    omega_new = np.empty_like(omega)
    coeff = ht / c
    reaction = 2.0 * alpha / r0
    hx2 = hx * hx

    # внутренние узлы
    for i in range(1, I):
        lapl = (omega[i + 1] - 2.0 * omega[i] + omega[i - 1]) / hx2
        omega_new[i] = omega[i] + coeff * (kappa * lapl - reaction * omega[i])

    # граничное соотношение на новом слое
    omega_new[0] = 0.5 * (omega_new[1] + omega_new[I - 1])
    omega_new[I] = omega_new[0]
    return omega_new

# РЕШЕНИЯ С ЭКОНОМИЕЙ ПАМЯТИ

def solve_explicit_sampled_time(c, kappa, alpha, r0, L, T, Nx, Nt, psi_func, x_points, progress=None):
    """
    Решение схемы и возврат временных рядов omega(t) для узлов, соответствующих x_points.
    x_points могут быть заданы как дроби [0,1] (доли L) или как абсолютные координаты.
    Возвращает (t_vals, results) где results.shape = (Nt+1, m).
    """
    hx = L / Nx
    ht = T / Nt
    x_grid = np.linspace(0.0, L, Nx + 1)

    # индексы узлов, соответствующие x_points
    idxs = []
    for x0 in x_points:
        # распознать: если значение в [0,1] трактуем как долю длины
        if 0.0 <= x0 <= 1.0:
            x_abs = x0 * L
        else:
            x_abs = x0
        i0 = int(round(x_abs / L * Nx))
        i0 = min(max(i0, 0), Nx)
        idxs.append(i0)

    m = len(idxs)
    results = np.zeros((Nt + 1, m))

    # начальное условие
    omega = psi_func(x_grid)
    omega[-1] = omega[0]

    # записать начальные значения
    for j, i0 in enumerate(idxs):
        results[0, j] = omega[i0]

    # основной цикл
    update_step = max(1, Nt // 100)
    for k in range(1, Nt + 1):
        omega = explicit_step(omega, c, kappa, alpha, r0, hx, ht)
        for j, i0 in enumerate(idxs):
            results[k, j] = omega[i0]

        if progress and (k % update_step == 0 or k == Nt):
            progress['value'] = k / Nt * 100
            progress.update()

    if progress:
        progress['value'] = 0
    t_vals = np.linspace(0.0, T, Nt + 1)
    return t_vals, results

def solve_explicit_sampled_space(c, kappa, alpha, r0, L, T, Nx, Nt, psi_func, t_points, progress=None):
    """
    Решение схемы и возврат пространственных профилей omega(x) для моментов t_points.
    t_points в тех же единицах времени, возвращает (x_vals, results) где results.shape = (m, Nx+1).
    """
    hx = L / Nx
    ht = T / Nt
    x_grid = np.linspace(0.0, L, Nx + 1)

    # индексы временных слоёв
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
    results = np.zeros((m, Nx + 1))

    # начальное условие
    omega = psi_func(x_grid)
    omega[-1] = omega[0]

    # сохранить при k=0, если нужно
    for j, k0 in enumerate(k_idxs):
        if k0 == 0:
            results[j, :] = omega.copy()

    update_step = max(1, Nt // 100)
    for k in range(1, Nt + 1):
        omega = explicit_step(omega, c, kappa, alpha, r0, hx, ht)
        for j, k0 in enumerate(k_idxs):
            if k == k0:
                results[j, :] = omega.copy()

        if progress and (k % update_step == 0 or k == Nt):
            progress['value'] = k / Nt * 100
            progress.update()

    if progress:
        progress['value'] = 0
    x_vals = x_grid
    return x_vals, results

# =================== ОТОБРАЖЕНИЕ (как в вашей программе) ===================

def plot_temp_vs_time(x_list, R, kappa, c, alpha, r0, ax, Nx, Nt, T, progress=None, psi_func=None):
    L = 2 * math.pi * R
    if psi_func is None:
        psi_func = lambda x: np.sin(2.0 * np.pi * x / L)  # аналогично вашему примеру
    # преобразуем x_list: автоматически проинтерпретируем дроби [0,1] как доли длины
    x_points = [float(x) for x in x_list]
    t_vals, results = solve_explicit_sampled_time(c, kappa, alpha, r0, L, T, Nx, Nt, psi_func, x_points, progress)
    ax.clear()
    for idx, x0 in enumerate(x_list):
        ax.plot(t_vals, results[:, idx], label=f"x₀ = {float(x0):.2f}")
    ax.set_title("Температура w(x₀, t)")
    ax.set_xlabel("Время t")
    ax.set_ylabel("w(x₀, t)")
    ax.legend()
    ax.grid(True)

def plot_temp_vs_x(t_list, R, kappa, c, alpha, r0, ax, Nx, Nt, T, progress=None, psi_func=None):
    L = 2 * math.pi * R
    if psi_func is None:
        psi_func = lambda x: np.sin(2.0 * np.pi * x / L)
    t_points = [float(t) for t in t_list]
    x_vals, results = solve_explicit_sampled_space(c, kappa, alpha, r0, L, T, Nx, Nt, psi_func, t_points, progress)
    ax.clear()
    for idx, t0 in enumerate(t_list):
        ax.plot(x_vals, results[idx, :], label=f"t = {float(t0):.2f}")
    ax.set_title("Распределение температуры w(x, t₀)")
    ax.set_xlabel("Координата x")
    ax.set_ylabel("w(x, t₀)")
    ax.legend()
    ax.grid(True)

# =================== GUI ===================

root = tk.Tk()
root.title("Температурный анализ (явная схема)")
root.geometry("1200x680")

main_frame = ttk.Frame(root)
main_frame.pack(fill="both", expand=True)

plot_frame = ttk.Frame(main_frame)
plot_frame.pack(side="left", fill="both", expand=True)

fig, ax = plt.subplots(figsize=(7, 5))
canvas = FigureCanvasTkAgg(fig, master=plot_frame)
canvas.get_tk_widget().pack(fill="both", expand=True)

control_frame = ttk.Frame(main_frame, padding=10)
control_frame.pack(side="right", fill="y")

mode_options = {"w(x₀, t)": "time", "w(x, t₀)": "space"}
mode_var = tk.StringVar(value="time")
x_values_default = "0, 0.1, 2, 5, 10"  # можно вводить доли или абсолютные
t_values_default = "0, 0.1, 0.5, 5, 10, 20"

progress_bar = ttk.Progressbar(control_frame, length=200)
progress_bar.pack(pady=(5, 10))

# Виджеты управления
ttk.Label(control_frame, text="Режим графика:").pack(anchor="w", pady=(0, 3))
mode_menu = ttk.Combobox(control_frame, values=list(mode_options.keys()), state="readonly")
mode_menu.current(0)
mode_menu.pack(anchor="w", fill="x")

ttk.Label(control_frame, text="Значения x₀ или t₀ (через запятую):").pack(anchor="w", pady=(10, 3))
values_entry = ttk.Entry(control_frame, width=35)
values_entry.insert(0, x_values_default)
values_entry.pack(anchor="w")

use_fraction_note = ttk.Label(control_frame, text="(Ввод: число в [0,1] = доля длины L; иначе абсолютная координата)")
use_fraction_note.pack(anchor="w", pady=(3, 6))

def update_mode(event=None):
    mode = mode_menu.get()
    if mode.startswith("w(x₀"):
        values_entry.delete(0, tk.END)
        values_entry.insert(0, x_values_default)
    else:
        values_entry.delete(0, tk.END)
        values_entry.insert(0, t_values_default)

mode_menu.bind("<<ComboboxSelected>>", lambda e: update_mode())

ttk.Button(control_frame, text="Построить график", command=lambda: run_plot()).pack(pady=10)

ttk.Button(control_frame, text="Настроить параметры", command=lambda: toggle_constants()).pack(pady=(5, 8))

# Параметры
constants_frame = ttk.Frame(control_frame)

def add_const_input(label, default):
    ttk.Label(constants_frame, text=label).pack(anchor="w", pady=(8, 2))
    entry = ttk.Entry(constants_frame, width=15)
    entry.insert(0, default)
    entry.pack(anchor="w")
    return entry

R_entry = add_const_input("Радиус R:", "4")
alpha_entry = add_const_input("α:", "0.003")
k_entry = add_const_input("k:", "0.45")
c_entry = add_const_input("c:", "1.3")
r0_entry = add_const_input("r₀:", "4.0")
Nx_entry = add_const_input("Число интервалов Nx:", "200")
Nt_entry = add_const_input("Число шагов по времени Nt:", "2000")
T_entry = add_const_input("Время моделирования T:", "1.0")

def toggle_constants():
    if constants_frame.winfo_ismapped():
        constants_frame.pack_forget()
    else:
        constants_frame.pack(fill="x", pady=(10, 0))

# Основной обработчик построения
def run_plot():
    try:
        mode = mode_menu.get()
        raw_vals = values_entry.get().split(",")
        vals = [float(s.strip()) for s in raw_vals if s.strip() != ""]

        # параметры
        R = float(R_entry.get())
        alpha = float(alpha_entry.get())
        kappa = float(k_entry.get())
        c = float(c_entry.get())
        r0 = float(r0_entry.get())
        Nx = int(Nx_entry.get())
        Nt = int(Nt_entry.get())
        T = float(T_entry.get())

        if Nx < 2 or Nt < 1:
            messagebox.showerror("Ошибка", "Nx должен быть >=2, Nt >=1")
            return

        L = 2.0 * math.pi * R

        # CFL-предупреждение
        hx = L / Nx
        cfl_limit = c * hx * hx / (2.0 * kappa) if kappa != 0 else float("inf")
        if T / Nt > cfl_limit:
            messagebox.showwarning("Предупреждение устойчивости",
                                   f"Шаг по времени h_t = {T/Nt:.3e} может превышать CFL-ограничение {cfl_limit:.3e}.\n"
                                   "Явная схема может быть неустойчива. Уменьшите h_t или увеличьте Nx.")

        # Вызов расчёта и построения
        if mode.startswith("w(x₀"):
            # x0 могут быть даны как доли (0..1) или как абсолютные координаты
            x_inputs = []
            for s in vals:
                x_inputs.append(s)
            plot_temp_vs_time(x_inputs, R, kappa, c, alpha, r0, ax, Nx, Nt, T, progress=progress_bar)
        else:
            t_inputs = []
            for s in vals:
                t_inputs.append(s)
            plot_temp_vs_x(t_inputs, R, kappa, c, alpha, r0, ax, Nx, Nt, T, progress=progress_bar)

        canvas.draw()
    except Exception as e:
        messagebox.showerror("Ошибка", f"Проверьте ввод данных:\n{e}")

update_mode()

root.mainloop()
