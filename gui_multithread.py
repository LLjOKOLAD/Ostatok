import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue


# Основные математические функции
def make_l(R):
    return 2 * math.pi * R


def coef_n2(k, c, R):
    l = make_l(R)
    return k / c * (2 * math.pi / l) ** 2


def coef_const(alpha, c, R):
    return (2 * alpha) / (c * R)


def Fi(n, t, x, R, k, c, alpha):
    l = make_l(R)
    if n == 0:
        return 0.5 * math.exp(-coef_const(alpha, c, R) * t)
    Bn = -((-1) ** n - 1) / (math.pi * n)
    omega_n = 2 * math.pi * n / l
    return Bn * math.sin(omega_n * x) * math.exp(-t * (coef_n2(k, c, R) * n ** 2 + coef_const(alpha, c, R)))


def num_of_iter(eps, t, R, k, c, alpha, MAX_N=10000):
    if t == 0:
        return MAX_N
    n = 1

    while True:
        Bn = abs((-((-1) ** n - 1)) / (math.pi * n))
        exp_factor = math.exp(-t * (coef_n2(k, c, R) * n ** 2 + coef_const(alpha, c, R)))
        term = Bn * exp_factor
        if term <= eps:
            break
        n += 2
        if n > MAX_N:
            break
    return n


class PlotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Температурный анализ")
        self.root.geometry("1200x680")

        self.setup_ui()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.running = False
        self.progress_queue = queue.Queue()

    def setup_ui(self):
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill="both", expand=True)

        self.plot_frame = ttk.Frame(self.main_frame)
        self.plot_frame.pack(side="left", fill="both", expand=True)

        self.fig, self.ax = plt.subplots(figsize=(7, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.control_frame = ttk.Frame(self.main_frame, padding=10)
        self.control_frame.pack(side="right", fill="y")

        self.mode_options = {"w(x₀, t)": "time", "w(x, t₀)": "space"}
        self.x_values = "0, 1, 2, 3, 4"  # Упорядоченные значения по умолчанию
        self.t_values = "0, 1, 5, 10, 20"  # Упорядоченные значения по умолчанию
        self.use_eps = tk.BooleanVar(value=False)

        self.create_widgets()
        self.update_mode()

    def create_widgets(self):
        # Выбор режима
        ttk.Label(self.control_frame, text="Режим графика:").pack(anchor="w", pady=(0, 3))
        self.mode_menu = ttk.Combobox(self.control_frame,
                                      values=list(self.mode_options.keys()),
                                      state="readonly")
        self.mode_menu.current(0)
        self.mode_menu.pack(anchor="w", fill="x")
        self.mode_menu.bind("<<ComboboxSelected>>", self.update_mode)

        # Ввод значений
        ttk.Label(self.control_frame, text="Значения (через запятую):").pack(anchor="w", pady=(10, 3))
        self.values_entry = ttk.Entry(self.control_frame, width=35)
        self.values_entry.insert(0, self.x_values)
        self.values_entry.pack(anchor="w")

        # Настройки точности
        ttk.Checkbutton(self.control_frame, text="Автоподбор N по точности ε",
                        variable=self.use_eps, command=self.toggle_eps).pack(anchor="w", pady=(10, 3))

        ttk.Label(self.control_frame, text="Число членов ряда N:").pack(anchor="w")
        self.N_entry = ttk.Entry(self.control_frame, width=15)
        self.N_entry.insert(0, "100")
        self.N_entry.pack(anchor="w")

        ttk.Label(self.control_frame, text="Точность ε:").pack(anchor="w")
        self.eps_entry = ttk.Entry(self.control_frame, width=15)
        self.eps_entry.insert(0, "1e-5")
        self.eps_entry.pack(anchor="w")

        # Прогресс-бар
        self.progress_frame = ttk.Frame(self.control_frame)
        self.progress_frame.pack(fill="x", pady=(10, 5))

        self.progress_label = ttk.Label(self.progress_frame, text="Прогресс: 0%")
        self.progress_label.pack(side="left")

        self.progress_bar = ttk.Progressbar(self.progress_frame, length=180)
        self.progress_bar.pack(side="left", padx=(5, 0))

        # Кнопки управления
        ttk.Button(self.control_frame, text="Построить график",
                   command=self.start_plot).pack(pady=(5, 10))

        # Параметры системы
        self.constants_frame = ttk.Frame(self.control_frame)
        self.create_constants_inputs()
        self.toggle_eps()

    def create_constants_inputs(self):
        ttk.Label(self.constants_frame, text="Радиус R:").pack(anchor="w", pady=(8, 2))
        self.R_entry = ttk.Entry(self.constants_frame, width=15)
        self.R_entry.insert(0, "1.0")
        self.R_entry.pack(anchor="w")

        ttk.Label(self.constants_frame, text="Коэффициент α:").pack(anchor="w", pady=(8, 2))
        self.alpha_entry = ttk.Entry(self.constants_frame, width=15)
        self.alpha_entry.insert(0, "0.01")
        self.alpha_entry.pack(anchor="w")

        ttk.Label(self.constants_frame, text="Коэффициент k:").pack(anchor="w", pady=(8, 2))
        self.k_entry = ttk.Entry(self.constants_frame, width=15)
        self.k_entry.insert(0, "0.5")
        self.k_entry.pack(anchor="w")

        ttk.Label(self.constants_frame, text="Коэффициент c:").pack(anchor="w", pady=(8, 2))
        self.c_entry = ttk.Entry(self.constants_frame, width=15)
        self.c_entry.insert(0, "1.0")
        self.c_entry.pack(anchor="w")

        self.constants_frame.pack(fill="x", pady=(10, 0))

    def update_mode(self, event=None):
        mode = self.mode_options[self.mode_menu.get()]
        if mode == "time":
            self.values_entry.delete(0, tk.END)
            self.values_entry.insert(0, self.x_values)
        else:
            self.values_entry.delete(0, tk.END)
            self.values_entry.insert(0, self.t_values)

    def toggle_eps(self):
        if self.use_eps.get():
            self.N_entry.config(state="disabled")
            self.eps_entry.config(state="normal")
        else:
            self.N_entry.config(state="normal")
            self.eps_entry.config(state="disabled")

    def start_plot(self):
        if self.running:
            return

        self.running = True
        self.progress_bar['value'] = 0
        self.progress_label.config(text="Прогресс: 0%")

        threading.Thread(target=self.run_calculation, daemon=True).start()
        self.root.after(100, self.check_progress)

    def check_progress(self):
        try:
            while True:
                progress = self.progress_queue.get_nowait()
                self.progress_bar['value'] = progress
                self.progress_label.config(text=f"Прогресс: {progress:.0f}%")
        except queue.Empty:
            pass

        if self.running:
            self.root.after(100, self.check_progress)

    def run_calculation(self):
        try:
            mode = self.mode_options[self.mode_menu.get()]
            values_str = self.values_entry.get().strip()

            # Парсим и сортируем значения
            values = sorted([float(x.strip()) for x in values_str.split(',') if x.strip()])

            R = float(self.R_entry.get())
            alpha = float(self.alpha_entry.get())
            k = float(self.k_entry.get())
            c = float(self.c_entry.get())

            N = None if self.use_eps.get() else int(self.N_entry.get())
            eps = float(self.eps_entry.get()) if self.use_eps.get() else None

            if mode == "time":
                t_vals = np.linspace(0, 20, 300)
                results = self.calculate_series(values, t_vals, R, k, c, alpha, N, eps, mode)
                self.root.after(0, lambda: self.draw_time_plot(t_vals, results))
            else:
                x_vals = np.linspace(0, make_l(R), 300)
                results = self.calculate_series(values, x_vals, R, k, c, alpha, N, eps, mode)
                self.root.after(0, lambda: self.draw_space_plot(x_vals, results))

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Ошибка", str(e)))
        finally:
            self.running = False

    def calculate_series(self, points, domain, R, k, c, alpha, N, eps, mode):
        results = {}
        total = len(points)

        with ThreadPoolExecutor() as executor:
            futures = {}
            for i, point in enumerate(points):
                if mode == "time":
                    args = [(point, t, R, k, c, alpha, N, eps) for t in domain]
                else:
                    args = [(x, point, R, k, c, alpha, N, eps) for x in domain]

                futures[executor.submit(self.calculate_curve, args)] = (i, point)

            for future in as_completed(futures):
                i, point = futures[future]
                results[point] = future.result()
                self.progress_queue.put((i + 1) / total * 100)

        return results

    def calculate_curve(self, args):
        return [self.calculate_point(*arg) for arg in args]

    def calculate_point(self, x, t, R, k, c, alpha, N=None, eps=None):
        if eps is not None:
            N = num_of_iter(eps, t, R, k, c, alpha)

        result = Fi(0, t, x, R, k, c, alpha)
        for n in range(1, N + 1):
            result += Fi(n, t, x, R, k, c, alpha)
        return result

    def draw_time_plot(self, t_vals, results):
        self.ax.clear()

        # Рисуем кривые в правильном порядке
        for x in sorted(results.keys()):
            self.ax.plot(t_vals, results[x], label=f"x = {x:.2f}")

        self.ax.set_title("Зависимость температуры от времени")
        self.ax.set_xlabel("Время t")
        self.ax.set_ylabel("Температура w(x, t)")
        self.ax.legend()
        self.ax.grid(True)

        self.canvas.draw()

    def draw_space_plot(self, x_vals, results):
        self.ax.clear()

        # Рисуем кривые в правильном порядке
        for t in sorted(results.keys()):
            self.ax.plot(x_vals, results[t], label=f"t = {t:.2f}")

        self.ax.set_title("Распределение температуры по координате")
        self.ax.set_xlabel("Координата x")
        self.ax.set_ylabel("Температура w(x, t)")
        self.ax.legend()
        self.ax.grid(True)

        self.canvas.draw()


if __name__ == '__main__':
    root = tk.Tk()
    app = PlotApp(root)
    root.mainloop()