import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import math


class ExplicitSchemeGUI:
    def __init__(self):
        # Параметры по умолчанию
        self.alpha = 0.003
        self.R = 4.0
        self.T = 30.0
        self.I = 50
        self.K = 1500
        self.k = 0.45
        self.c = 1.3
        self.r0 = 4.0

        self.matrix = None
        self.mas_x = None
        self.mas_t = None
        self.hr = 0
        self.ht = 0

        # Главное окно
        self.root = tk.Tk()
        self.root.title("Температурный анализ (Явная схема)")
        self.root.geometry("1200x680")

        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True)

        # График
        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(side="left", fill="both", expand=True)

        self.fig, self.ax = plt.subplots(figsize=(7, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Панель управления
        control_frame = ttk.Frame(main_frame, padding=10)
        control_frame.pack(side="right", fill="y")

        # Прогрессбар
        self.progress = ttk.Progressbar(control_frame, length=200)
        self.progress.pack(pady=(5, 10))

        # Режим
        mode_options = {"ω(x₀, t)": "time", "ω(x, t₀)": "space"}
        self.mode_var = tk.StringVar(value="time")

        ttk.Label(control_frame, text="Режим графика:").pack(anchor="w", pady=(0, 3))
        self.mode_menu = ttk.Combobox(control_frame, values=list(mode_options.keys()), state="readonly")
        self.mode_menu.current(0)
        self.mode_menu.pack(anchor="w", fill="x")
        self.mode_menu.bind("<<ComboboxSelected>>",
                            lambda e: self.mode_var.set(mode_options[self.mode_menu.get()]) or self.update_mode())

        # Значения x₀ / t₀
        ttk.Label(control_frame, text="Значения x₀ или t₀ (через запятую):").pack(anchor="w", pady=(10, 3))
        self.values_entry = ttk.Entry(control_frame, width=35)
        self.values_entry.insert(0, "0, 0.5, 1, 2, 3")
        self.values_entry.pack(anchor="w")

        # Кнопки
        ttk.Button(control_frame, text="Построить график", command=self.run_plot).pack(pady=10)
        ttk.Button(control_frame, text="Пересчитать матрицу", command=self.recalculate_matrix).pack(pady=(5, 10))
        ttk.Button(control_frame, text="Настроить параметры", command=self.toggle_constants).pack(pady=(5, 10))
        ttk.Button(control_frame, text="Очистить", command=self.clear_plot).pack(pady=(5, 10))

        # Блок параметров
        self.constants_frame = ttk.Frame(control_frame)
        self._add_parameters()

        self.update_mode()
        self.root.mainloop()

    # ---------------- Вспомогательные ----------------

    def _add_parameters(self):
        def add_field(label, default):
            ttk.Label(self.constants_frame, text=label).pack(anchor="w", pady=(5, 0))
            e = ttk.Entry(self.constants_frame, width=15)
            e.insert(0, str(default))
            e.pack(anchor="w")
            return e

        self.R_entry = add_field("Радиус R:", self.R)
        self.alpha_entry = add_field("α:", self.alpha)
        self.k_entry = add_field("k:", self.k)
        self.c_entry = add_field("c:", self.c)
        self.I_entry = add_field("Число узлов I:", self.I)
        self.K_entry = add_field("Шагов по времени K:", self.K)
        self.T_entry = add_field("Время моделирования T:", self.T)

    def update_mode(self):
        if self.mode_var.get() == "time":
            self.values_entry.delete(0, tk.END)
            self.values_entry.insert(0, "0, 0.1, 2, 5, 10")
        else:
            self.values_entry.delete(0, tk.END)
            self.values_entry.insert(0, "0, 0.1, 0.5, 5, 10, 20")

    def toggle_constants(self):
        if self.constants_frame.winfo_ismapped():
            self.constants_frame.pack_forget()
        else:
            self.constants_frame.pack(fill="x", pady=(10, 0))

    # ---------------- Математическая часть ----------------

    def phi_r(self, x, R):
        """psi_2(x): 1 on [0, pi*R], 0 on (pi*R, 2*pi*R)"""
        L = 2.0 * math.pi * R
        half = math.pi * R  # = L/2
        # x can be array or scalar
        return np.where(x <= half, 1.0, 0.0)

    def recalculate_matrix(self):
        """Полный пересчёт матрицы"""
        try:
            self.R = float(self.R_entry.get())
            self.alpha = float(self.alpha_entry.get())
            self.k = float(self.k_entry.get())
            self.c = float(self.c_entry.get())
            self.r0 = float(self.R_entry.get())
            self.I = int(self.I_entry.get())
            self.K = int(self.K_entry.get())
            self.T = float(self.T_entry.get())

            L = 2 * math.pi * self.R
            self.hr = L / self.I
            self.ht = self.T / self.K
            self.mas_x = np.linspace(0, L, self.I + 1)
            self.mas_t = np.linspace(0, self.T, self.K + 1)
            self.matrix = np.zeros((self.K + 1, self.I + 1))

            self.matrix[0, :] = self.phi_r(self.mas_x,self.R)
            self.matrix[0, -1] = self.matrix[0, 0]

            reaction = 2 * self.alpha / self.r0
            hx2 = self.hr ** 2

            update_step = max(1, self.K // 100)
            for k in range(1, self.K + 1):
                prev = self.matrix[k - 1]
                cur = self.matrix[k]
                for i in range(1, self.I):
                    lapl = (prev[i + 1] - 2 * prev[i] + prev[i - 1]) / hx2
                    cur[i] = prev[i] + (self.ht / self.c) * (self.k * lapl - reaction * prev[i])

                cur[0] = 0.5 * (cur[1] + cur[self.I - 1])
                cur[self.I] = cur[0]

                if k % update_step == 0:
                    self.progress['value'] = k / self.K * 100
                    self.root.update_idletasks()

            self.progress['value'] = 0
            messagebox.showinfo("Готово", "Матрица успешно пересчитана.")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при вычислении: {e}")

    def run_plot(self):
        """Построение графика"""
        if self.matrix is None:
            messagebox.showwarning("Нет данных", "Сначала пересчитайте матрицу.")
            return

        try:
            vals = [float(v.strip()) for v in self.values_entry.get().split(",") if v.strip()]
            mode = self.mode_var.get()

            self.ax.clear()
            if mode == "time":
                for x0 in vals:
                    idx = np.searchsorted(self.mas_x, x0)
                    idx = min(idx, self.I)
                    self.ax.plot(self.mas_t, self.matrix[:, idx], label=f"x₀={x0:.2f}")
                self.ax.set_title("Температура ω(x₀, t)")
                self.ax.set_xlabel("Время t, с")
                self.ax.set_ylabel("ω(x₀, t)")
            else:
                for t0 in vals:
                    idx = np.searchsorted(self.mas_t, t0)
                    idx = min(idx, self.K)
                    self.ax.plot(self.mas_x, self.matrix[idx, :], label=f"t={t0:.2f}")
                self.ax.set_title("Распределение ω(x, t₀)")
                self.ax.set_xlabel("Координата x")
                self.ax.set_ylabel("ω(x, t₀)")

            self.ax.legend()
            self.ax.grid(True)
            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при построении графика:\n{e}")

    def clear_plot(self):
        self.ax.clear()
        self.canvas.draw()


if __name__ == "__main__":
    ExplicitSchemeGUI()
