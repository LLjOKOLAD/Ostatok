import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from fontTools.cu2qu.cu2qu import MAX_N
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import math

# Расчёты
# Длина кольца

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
    Bn = -((-1)**n - 1) / (math.pi * n)
    omega_n = 2 * math.pi * n / l
    return Bn * np.sin(omega_n * x) * np.exp(-t * (coef_n2(k, c, R) * n ** 2 + coef_const(alpha, c, R)))

# Частичная сумма ряда
def SUM(x, t, N, R, k, c, alpha):
    result = Fi(0, t, x, R, k, c, alpha)
    for n in range(1, N + 1):
        result += Fi(n, t, x, R, k, c, alpha)
    return result


# Подбор N по модулю одного члена ряда
def num_of_iter(eps, t, x, R, k, c, alpha):
    MAX_N = 5000
    if t == 0:
        # При t=0 подбор N невозможен, ставим большое значение по умолчанию
        return MAX_N
    n = 1
    while True:
        l = make_l(R)
        Bn = abs((-((-1)**n - 1)) / (math.pi * n))
        exp_factor = math.exp(-t * (coef_n2(k, c, R) * n**2 + coef_const(alpha, c, R)))
        term = Bn * exp_factor
        if term <= eps:
            break
        n += 2  # только нечётные n
        if n > MAX_N:
            break
    return n

# Уточнённый подбор N по разности сумм
def num_of_iter_exp(eps, t, x, R, k, c, alpha):
    Neps = num_of_iter(eps, t, x, R, k, c, alpha)
    Nexp = Neps
    while abs(SUM(x, t, Neps, R, k, c, alpha) - SUM(x, t, Nexp, R, k, c, alpha)) <= eps and Nexp > 0:
        Nexp -= 2
    return Nexp + 2

# Общая функция частичной суммы
def partial_sum(x, t, R, k, c, alpha, N=None, eps=None):
    MIN_N = 5  # Минимальное количество членов для адекватного приближения
    if eps is not None:
        N = max(num_of_iter(eps, t, x, R, k, c, alpha), MIN_N)
    result = Fi(0, t, x, R, k, c, alpha)
    for n in range(1, N + 1):
        result += Fi(n, t, x, R, k, c, alpha)
    return result

# Построение графика зависимости температуры от времени
def plot_temp_vs_time(x_list, R, k, c, alpha, ax, N=None, eps=None, progress=None):
    t_vals = np.linspace(0, 30, 300)
    ax.clear()
    for idx, x0 in enumerate(x_list):
        u_vals = [partial_sum(x0, t, R, k, c, alpha, N, eps) for t in t_vals]
        ax.plot(t_vals, u_vals, label=f"x₀ = {x0:.2f}")
        if progress:
            progress['value'] = (idx + 1) / len(x_list) * 100
            progress.update()
    ax.set_title("Температура w(x₀, t)")
    ax.set_xlabel("Время t")
    ax.set_ylabel("w(x₀, t)")
    ax.legend()
    ax.grid(True)

# Построение графика распределения температуры по координате
def plot_temp_vs_x(t_list, R, k, c, alpha, ax, N=None, eps=None, progress=None):
    x_vals = np.linspace(0, make_l(R), 300)
    ax.clear()
    for idx, t0 in enumerate(t_list):
        u_vals = [partial_sum(x, t0, R, k, c, alpha, N, eps) for x in x_vals]
        ax.plot(x_vals, u_vals, label=f"t = {t0:.2f}")
        if progress:
            progress['value'] = (idx + 1) / len(t_list) * 100
            progress.update()
    ax.set_title("Распределение температуры w(x, t₀)")
    ax.set_xlabel("Координата x")
    ax.set_ylabel("w(x, t₀)")
    ax.legend()
    ax.grid(True)



# Основное окно
root = tk.Tk()
root.title("Температурный анализ")
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
x_values = "0, 0.1, 2, 5, 10"
t_values = "0, 0.1, 0.5, 5, 10, 20"

use_eps = tk.BooleanVar(value=False)

progress_bar = ttk.Progressbar(control_frame, length=200)
progress_bar.pack(pady=(5, 10))

N_entry = ttk.Entry(control_frame, width=15)
N_entry.insert(0, "100")

eps_entry = ttk.Entry(control_frame, width=15)
eps_entry.insert(0, "1e-5")

def update_mode(event=None):
    mode = mode_var.get()
    if mode == "time":
        values_entry.delete(0, tk.END)
        values_entry.insert(0, x_values)
    else:
        values_entry.delete(0, tk.END)
        values_entry.insert(0, t_values)

def toggle_eps():
    if use_eps.get():
        N_entry.config(state="disabled")
        eps_entry.config(state="normal")
    else:
        N_entry.config(state="normal")
        eps_entry.config(state="disabled")

def run_plot():
    global x_values, t_values
    try:
        mode = mode_var.get()
        vals = list(map(float, values_entry.get().split(',')))
        if mode == "time":
            x_values = values_entry.get()
        else:
            t_values = values_entry.get()

        R = float(R_entry.get())
        alpha = float(alpha_entry.get())
        k = float(k_entry.get())
        c = float(c_entry.get())

        N = None if use_eps.get() else int(N_entry.get())
        eps = float(eps_entry.get()) if use_eps.get() else None

        if mode == "time":
            plot_temp_vs_time(vals, R, k, c, alpha, ax, N, eps, progress_bar)
        else:
            plot_temp_vs_x(vals, R, k, c, alpha, ax, N, eps, progress_bar)

        canvas.draw()
        progress_bar['value'] = 0
    except Exception as e:
        messagebox.showerror("Ошибка", f"Проверь ввод данных:\n{e}")

def toggle_constants():
    if constants_frame.winfo_ismapped():
        constants_frame.pack_forget()
    else:
        constants_frame.pack(fill="x", pady=(10, 0))

# Виджеты

ttk.Label(control_frame, text="Режим графика:").pack(anchor="w", pady=(0, 3))
mode_menu = ttk.Combobox(control_frame, values=list(mode_options.keys()), state="readonly")
mode_menu.current(0)
mode_menu.pack(anchor="w", fill="x")
mode_menu.bind("<<ComboboxSelected>>", lambda e: mode_var.set(mode_options[mode_menu.get()]) or update_mode())

ttk.Label(control_frame, text="Значения x₀ или t₀ (через запятую):").pack(anchor="w", pady=(10, 3))
values_entry = ttk.Entry(control_frame, width=35)
values_entry.insert(0, x_values)
values_entry.pack(anchor="w")

mode_frame = ttk.Frame(control_frame)
mode_frame.pack(anchor="w", pady=(10, 3))
ttk.Checkbutton(mode_frame, text="Использовать ε вместо N", variable=use_eps, command=toggle_eps).pack(anchor="w")

ttk.Label(control_frame, text="Число членов ряда N:").pack(anchor="w")
N_entry.pack(anchor="w")

ttk.Label(control_frame, text="Точность ε:").pack(anchor="w")
eps_entry.pack(anchor="w")

ttk.Button(control_frame, text="Построить график", command=run_plot).pack(pady=20)
ttk.Button(control_frame, text="Настроить параметры", command=toggle_constants).pack(pady=(10, 5))

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

update_mode()
toggle_eps()

root.mainloop()