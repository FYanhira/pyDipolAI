import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import mean_squared_error, r2_score

from models.debye import DebyeModel  # Importa tu clase DebyeModel desde models/debye.py
from models.colecole import ColeColeModel
from models.coledavidson import ColeDavidsonModel
from models.havriliak_negami import HavriliakNegamiModel
from models.fractional_1cr import Fractional1CRModel
from models.fractional_2cr import Fractional2CRModel

def mean_absolute_deviation(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def calculate_fit_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    aad = mean_absolute_deviation(y_true, y_pred)
    return r2, mse, aad

class DielectricGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Fractional Dielectric Model Fitter")

        self.data = None
        self.freq = None
        self.param_inputs = {}
        self.result_texts = {}

        # Instanciar modelos disponibles
        self.registered_models = {
            'Debye': DebyeModel(),
            'Cole-Cole': ColeColeModel(),
            'Cole-Davidson': ColeDavidsonModel(),
            'Havriliak-Negami': HavriliakNegamiModel(),
            'Fracc-1CR': Fractional1CRModel(),
            'Fracc-2CR': Fractional2CRModel(),
        }

        # --- Nueva organización de controles superiores ---
        self.top_control_frame = tk.Frame(root)
        self.top_control_frame.pack(fill=tk.X, padx=10, pady=5)

        # Botón Load CSV (en columna 0)
        self.load_button = tk.Button(self.top_control_frame, text="Load CSV", command=self.load_csv)
        self.load_button.grid(row=0, column=0, padx=5)

        # Frame modelos (en columna 1)
        self.model_frame = tk.LabelFrame(self.top_control_frame, text="Select Models to Fit")
        self.model_frame.grid(row=0, column=1, sticky='nsw', padx=5)
        self.model_checks = {}
        for model_name in self.registered_models:
            var = tk.BooleanVar()
            chk = tk.Checkbutton(self.model_frame, text=model_name, variable=var, command=self.update_params)
            chk.pack(anchor='w')
            self.model_checks[model_name] = var

        # Frame parámetros (en columna 2)
        self.params_frame = tk.LabelFrame(self.top_control_frame, text="Model Parameters")
        self.params_frame.grid(row=0, column=2, sticky='nsew', padx=5)

        # Hacer que la columna 2 se expanda
        self.top_control_frame.grid_columnconfigure(2, weight=1)

        # Botón de ajuste (columna 0, fila 1)
        self.fit_button = tk.Button(self.top_control_frame, text="Fit Selected Models", command=self.run_fit)
        self.fit_button.grid(row=1, column=0, pady=(5, 0), sticky='w')

        # Frame de resultados (columna 3)
        self.result_frame = tk.LabelFrame(self.top_control_frame, text="Fit Results")
        self.result_frame.grid(row=0, column=3, rowspan=2, sticky='nsew', padx=5)
        self.top_control_frame.grid_columnconfigure(3, weight=1)

        # Frame para gráficas debajo de todo
        self.plot_frame = tk.Frame(root)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)


    def update_params(self):
        # Limpiar widgets anteriores
        for widget in self.params_frame.winfo_children():
            widget.destroy()
        self.param_inputs.clear()

        # Mostrar inputs de parámetros para cada modelo seleccionado
        for model_name, var in self.model_checks.items():
            if var.get():
                model_instance = self.registered_models[model_name]
                params = model_instance.get_params()

                label = tk.Label(self.params_frame, text=f"--- {model_name} Parameters ---", font=('Arial', 10, 'bold'))
                label.pack()

                self.param_inputs[model_name] = {}

                for key, (val, vmin, vmax) in params.items():
                    frame = tk.Frame(self.params_frame)
                    frame.pack()

                    tk.Label(frame, text=f"{key}: ").grid(row=0, column=0)
                    val_entry = tk.Entry(frame, width=7)
                    val_to_insert = f"{val:.4g}" if val is not None else "0.0"
                    val_entry.insert(0, val_to_insert)
                    val_entry.grid(row=0, column=1)

                    min_entry = tk.Entry(frame, width=7)
                    min_entry.insert(0, str(vmin))
                    min_entry.grid(row=0, column=2)

                    max_entry = tk.Entry(frame, width=7)
                    max_entry.insert(0, str(vmax))
                    max_entry.grid(row=0, column=3)

                    self.param_inputs[model_name][key] = {
                        'val': val_entry,
                        'min': min_entry,
                        'max': max_entry
                    }

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return
        df = pd.read_csv(file_path)
        try:
            self.freq = df['frequency'].values
            real = df['eps_real'].values
            imag = df['eps_imag'].values
            self.data = real + 1j * imag
            messagebox.showinfo("Success", "CSV loaded successfully.")

            # -------- NUEVO: actualizar parámetros automáticos --------
            
            for model_name, model_instance in self.registered_models.items():
                if hasattr(model_instance, 'set_auto_params_from_data'):
                    model_instance.set_auto_params_from_data(np.real(self.data))


            self.update_params()  # actualiza las cajas de entrada en la GUI

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {e}")

    def run_fit(self):
        if self.data is None or self.freq is None:
            messagebox.showwarning("Missing Data", "Please load data first.")
            return

        # Limpiar áreas de gráficos y resultados previos
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        for widget in self.result_frame.winfo_children():
            widget.destroy()

        fig, axs = plt.subplots(2, 2, figsize=(12, 10), dpi=100)
        axs = axs.flatten()

        eps_real = np.real(self.data)
        eps_imag = np.imag(self.data)
        tan_delta = eps_imag / eps_real

        axs[0].set_title("Parte Real ε′")
        axs[1].set_title("Parte Imaginaria ε″")
        axs[2].set_title("Tan δ = ε″ / ε′")
        axs[3].set_title("Gráfica de Cole-Cole: ε″ vs ε′")

        axs[0].semilogx(self.freq, eps_real, 'k.', label='Data')
        axs[1].semilogx(self.freq, eps_imag, 'k.', label='Data')
        axs[2].semilogx(self.freq, tan_delta, 'k.', label='Data')
        axs[3].plot(eps_real, eps_imag, 'ko', label='Data')

        for model_name, checked in self.model_checks.items():
            if checked.get():
                model_instance = self.registered_models[model_name]

                if model_name in self.param_inputs:
                    # Construir dict de parámetros desde inputs GUI
                    param_dict = {}
                    for key in self.param_inputs[model_name]:
                        try:
                            val_str = self.param_inputs[model_name][key]['val'].get()
                            min_str = self.param_inputs[model_name][key]['min'].get()
                            max_str = self.param_inputs[model_name][key]['max'].get()

                            val = float(val_str) if val_str not in ('', 'None') else 0.0
                            minval = float(min_str) if min_str not in ('', 'None') else 0.0
                            maxval = float(max_str) if max_str not in ('', 'None') else 1e6

                            param_dict[key] = {'val': val, 'min': minval, 'max': maxval}
                        except ValueError:
                            messagebox.showerror("Error", f"Invalid parameter input for {key} in model {model_name}.")
                            return

                else:
                    param_dict = None

                # Ejecutar ajuste
                result_real, result_imag = model_instance.fit(self.freq, eps_real, eps_imag, param_dict)

                eps_r_fit = result_real.best_fit
                eps_i_fit = result_imag.best_fit
                tan_d_fit = eps_i_fit / eps_r_fit

                axs[0].semilogx(self.freq, eps_r_fit, label=model_name)
                axs[1].semilogx(self.freq, eps_i_fit, label=model_name)
                axs[2].semilogx(self.freq, tan_d_fit, label=model_name)
                axs[3].plot(eps_r_fit, eps_i_fit, label=model_name)

                # Mostrar resultados en caja de texto
                result_box = tk.Text(self.result_frame, height=15, width=90)
                result_box.insert(tk.END, f"===== {model_name} =====\n\n")

                for pname, pval in result_imag.params.items():
                    result_box.insert(tk.END, f"{pname} = {pval.value:.6g}\n")

                r2_real, mse_real, aad_real = calculate_fit_metrics(eps_real, eps_r_fit)
                r2_imag, mse_imag, aad_imag = calculate_fit_metrics(eps_imag, eps_i_fit)

                result_box.insert(tk.END, f"\n--- Métricas Parte Real (ε′) ---\n")
                result_box.insert(tk.END, f"R² = {r2_real:.4f}, MSE = {mse_real:.4g}, AAD = {aad_real:.4g}\n")

                result_box.insert(tk.END, f"\n--- Métricas Parte Imaginaria (ε″) ---\n")
                result_box.insert(tk.END, f"R² = {r2_imag:.4f}, MSE = {mse_imag:.4g}, AAD = {aad_imag:.4g}\n")

                result_box.pack(pady=5)
                self.result_texts[model_name] = result_box

        for ax in axs:
            ax.legend()
            ax.grid(True)

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = DielectricGUI(root)
    root.mainloop()
