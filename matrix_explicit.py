import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import math
import threading


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
        ttk.Button(control_frame, text="Режим сходимости", command=self.open_convergence_mode).pack(pady=(10, 5))

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

# ----------------- Режим сходимости -----------------


    def open_convergence_mode(self):
        """
        Открывает окно для графического доказательства сходимости.
        (Ничего не запускается автоматически — вызывается по нажатию кнопки.)
        """
        conv_win = tk.Toplevel()
        conv_win.title("Графическое доказательство сходимости")
        conv_win.geometry("900x900")

        top_frame = ttk.Frame(conv_win, padding=6)
        top_frame.pack(fill="x")

        # выбор фиксированного параметра
        mode_var = tk.StringVar(value="r")
        ttk.Label(top_frame, text="Фиксированный параметр:").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(top_frame, text="r (фиксированный x)", variable=mode_var, value="r").grid(row=0, column=1, sticky="w")
        ttk.Radiobutton(top_frame, text="t (фиксированное время)", variable=mode_var, value="t").grid(row=0, column=2, sticky="w")

        ttk.Label(top_frame, text="Значение фиксированной переменной (доля L или абсолютная):").grid(row=1, column=0, columnspan=3, sticky="w", pady=(6,0))
        fixed_entry = ttk.Entry(top_frame, width=30)
        fixed_entry.insert(0, "0.5")
        fixed_entry.grid(row=2, column=0, columnspan=3, sticky="w")

        ttk.Label(top_frame, text="Пары (I,K) через точку с запятой (например: 20,500; 40,1000; 80,2000):").grid(row=3,columnspan=3,sticky="w",pady=(8, 0))
        pairs_entry = ttk.Entry(top_frame, width=60)
        pairs_entry.insert(0, "10,250; 20,500; 40,1000; 80,2000; 160,4000")
        pairs_entry.grid(row=4, column=0, columnspan=3, sticky="w")

        ttk.Label(top_frame, text="Количество точек для аналитики:").grid(row=5, column=0, sticky="w", pady=(6,0))
        analytic_N_entry = ttk.Entry(top_frame, width=10)
        analytic_N_entry.insert(0, "300")
        analytic_N_entry.grid(row=5, column=1, sticky="w")

        show_error_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(top_frame, text="Показать погрешность", variable=show_error_var).grid(row=6, column=0, columnspan=3, sticky="w", pady=(6, 0))

        # --- Область графика ---
        fig_conv, (ax_sol, ax_err) = plt.subplots(2, 1, figsize=(8, 5), sharex=True,gridspec_kw={'height_ratios': [3, 1]})
        fig_conv.subplots_adjust(hspace=0.25)
        canvas_conv = FigureCanvasTkAgg(fig_conv, master=conv_win)
        canvas_conv.get_tk_widget().pack(fill="both", expand=True, pady=6)

        progress = ttk.Progressbar(conv_win, length=400)
        progress.pack(pady=6)

        info_label = ttk.Label(conv_win, text="", foreground="red")
        info_label.pack()

        def explicit_solve_matrix(R, kappa, c, alpha, I, K, T):
            """
            Быстрое локальное решение явной схемы для заданных параметров.
            Возвращает (U, x_grid, t_grid), где U.shape = (K+1, I+1).
            Реализовано автономно — не меняет глобальную матрицу программы.
            """
            L = 2.0 * math.pi * R
            hr = L / I
            ht = T / K
            x_grid = np.linspace(0.0, L, I + 1)
            t_grid = np.linspace(0.0, T, K + 1)
            U = np.zeros((K + 1, I + 1), dtype=float)

            # начальное условие psi2 (ты сказал, что используешь psi2)
            # если у тебя другая psi, замени эту строку на вызов своей функции
            half = math.pi * R
            U[0, :] = np.where(x_grid <= half, 1.0, 0.0)
            U[0, -1] = U[0, 0]

            reaction = 2.0 * alpha / R
            hx2 = hr * hr

            # простая CFL-проверка: если ht слишком велико, алгоритм вернёт None
            if kappa != 0:
                ht_limit = (c * hx2) / (2.0 * kappa)
                if ht > ht_limit:
                    # признак несостоятельности; возвращаем None, чтобы вызывающий мог обработать
                    return None, x_grid, t_grid

            # шаг по времени
            for n in range(1, K + 1):
                prev = U[n - 1]
                cur = U[n]
                for j in range(1, I):
                    lap = (prev[j + 1] - 2.0 * prev[j] + prev[j - 1]) / hx2
                    cur[j] = prev[j] + (ht / c) * (kappa * lap - reaction * prev[j])
                cur[0] = 0.5 * (cur[1] + cur[I - 1])
                cur[I] = cur[0]
            return U, x_grid, t_grid

        def make_l(R):
            return 2 * math.pi * R

        def coef_n2(k, c, R):
            l = make_l(R)
            return k / c * (2 * math.pi / l) ** 2  # = k / (c * R^2)

        def coef_const(alpha, c, R):
            return (2 * alpha) / (c * R)

        def Fi(n, t, x, R, k, c, alpha):
            l = make_l(R)
            if n == 0:
                return 0.5 * math.exp(-coef_const(alpha, c, R) * t)
            Bn = -((-1) ** n - 1) / (math.pi * n)
            omega_n = 2 * math.pi * n / l
            return Bn * np.sin(omega_n * x) * np.exp(-t * (coef_n2(k, c, R) * n ** 2 + coef_const(alpha, c, R)))

        def FI(n, t, R, k, c, alpha):
            exponent = t * ((2 * alpha) / (c * R) + (k * n ** 2) / (c * R ** 2))
            denominator = (n + 1) ** 2 * k * math.pi * t * math.exp(exponent)
            return (c * R ** 2) / denominator

        def num_of_iter(eps, t, R, k, c, alpha):
            MAX_N = 5000
            if t == 0:
                # При t=0 подбор N невозможен, ставим большое значение по умолчанию
                return MAX_N
            n = 1
            while abs(FI(n, t, R, k, c, alpha)) >= eps:
                n += 1
                if n > MAX_N:
                    break
            return n

        def partial_sum(x, t, R, k, c, alpha, N=None, eps=None):
            MIN_N = 5  # Минимальное количество членов для адекватного приближения
            if eps is not None:
                N = max(num_of_iter(eps, t, R, k, c, alpha), MIN_N)
            result = Fi(0, t, x, R, k, c, alpha)
            for n in range(1, N + 1):
                result += Fi(n, t, x, R, k, c, alpha)
            return result

        def worker_build():
            # чтение параметров из главного окна
            try:
                # берем параметры из тех Entry, что уже есть в основной программе
                R = float(self.R_entry.get())
                kappa = float(self.k_entry.get())
                c = float(self.c_entry.get())
                alpha = float(self.alpha_entry.get())
                T = float(self.T_entry.get())
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось прочитать параметры из полей: {e}")
                return

            # --- Читаем пары ---
            try:
                pairs = []
                for p in pairs_entry.get().split(";"):
                    if p.strip():
                        I, K = map(int, p.strip().split(","))
                        pairs.append((I, K))
            except:
                messagebox.showerror("Ошибка", "Некорректный формат пар (I,K). Пример: 20,500; 40,1000; 80,2000")
                return

            # фиксированное значение
            try:
                fixed_val = float(fixed_entry.get())
            except:
                messagebox.showerror("Ошибка", "Неверное значение фиксированной переменной.")
                return

            analytic_N = int(analytic_N_entry.get())
            mode = mode_var.get()
            show_error = show_error_var.get()
            ax_sol.clear(); ax_err.clear()
            info_label.config(text="")

            # --- Аналитическое решение ---
            try:
                if mode == "r":
                    L = 2 * math.pi * R
                    x0 = fixed_val * L if 0 <= fixed_val <= 1 else fixed_val
                    t_vals = np.linspace(0, T, analytic_N)
                    analytic_vals = np.array([partial_sum(x0, t, R, kappa, c, alpha, 200) for t in t_vals])
                    ax_sol.plot(t_vals, analytic_vals, 'k-', lw=2, label="Аналитическое")
                    x_axis = t_vals
                else:
                    L = 2 * math.pi * R
                    t0 = fixed_val * T if 0 <= fixed_val <= 1 else fixed_val
                    x_vals = np.linspace(0, 2 * math.pi * R, analytic_N)
                    analytic_vals = np.array([partial_sum(x, t0, R, kappa, c, alpha,200) for x in x_vals])
                    ax_sol.plot(x_vals, analytic_vals, 'k-', lw=2, label="Аналитическое")
                    x_axis = x_vals
            except NameError:
                info_label.config(text="Функция SUM не найдена — добавь её.")
                return

            # --- Для каждой пары сеток ---
            total = len(pairs)
            errors = []
            labels = []
            for i, (I, K) in enumerate(pairs):
                progress['value'] = int((i + 1) / total * 100)
                progress.update()

                U, xg, tg = explicit_solve_matrix(R, kappa, c, alpha, I, K, T)
                if U is None: continue

                if mode == "r":
                    L = 2 * math.pi * R
                    x0 = fixed_val * L if 0 <= fixed_val <= 1 else fixed_val
                    idx = min(max(int(round(x0 / L * I)), 0), I)
                    num_vals = np.interp(x_axis, tg, U[:, idx])
                else:
                    t0 = fixed_val * T if 0 <= fixed_val <= 1 else fixed_val
                    idx = min(max(int(round(t0 / T * K)), 0), K)
                    num_vals = np.interp(x_axis, xg, U[idx, :])

                ax_sol.plot(x_axis, num_vals, label=f"I={I}, K={K}")
                err = np.abs(num_vals - analytic_vals)
                errors.append(err)
                labels.append(f"I={I}, K={K}")

            # --- График решения ---
            ax_sol.set_ylabel("ω(x,t)")
            ax_sol.set_title("Сходимость численного решения")
            ax_sol.grid(True)
            ax_sol.legend(fontsize='small', loc='best')

            # --- График ошибки ---
            if show_error and errors:
                for err, lbl in zip(errors, labels):
                    ax_err.plot(x_axis, err, label=lbl)
                ax_err.set_xlabel("t" if mode == "r" else "x")
                ax_err.set_ylabel("|Δ|")
                ax_err.set_title("Погрешность |ω_числ − ω_ан|")
                ax_err.grid(True)
                ax_err.legend(fontsize='x-small', loc='best')

            canvas_conv.draw()
            progress['value'] = 0
        # запуск в отдельном потоке, чтобы GUI не зависал на долгих расчетах
        def on_build_pressed():
            thread = threading.Thread(target=worker_build)
            thread.daemon = True
            thread.start()

        ttk.Button(conv_win, text="Построить график", command=on_build_pressed).pack(pady=6)
        ttk.Button(conv_win, text="Закрыть", command=conv_win.destroy).pack(pady=2)

    # ----------------- конец фрагмента режима сходимости -----------------


if __name__ == "__main__":
    ExplicitSchemeGUI()
