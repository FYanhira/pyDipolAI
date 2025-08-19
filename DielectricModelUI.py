import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter import scrolledtext  # <<--- agregado
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import mean_squared_error, r2_score

# ---- BO starter (as provided by you) ----
from optim.bo_starter import BayesianStarter

# ---- Models (as provided by you) ----
from models.debye import DebyeModel
from models.colecole import ColeColeModel
from models.coledavidson import ColeDavidsonModel
from models.havriliak_negami import HavriliakNegamiModel
from models.fractional_1cr import Fractional1CRModel
from models.fractional_2cr import Fractional2CRModel

from models.Debye_modulus import DebyeModulusModel
from models.Colecole_modulus import ColeColeModulusModel
from models.ColeDavidson_modulus import ColeDavidsonModulusModel
from models.HN_modulus import HNModulusModel
from models.fract_1cr_modulus import Fract1crModulusModel
from models.fract_2cr_modulus import Fract2crModulusModel

# ================== métricas ==================
def mean_absolute_deviation(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def calculate_fit_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    aad = mean_absolute_deviation(y_true, y_pred)
    return r2, mse, aad

# ================== GUI ==================
class DielectricGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Fractional Dielectric Model Fitter")
        self.data = None
        self.freq = None

        # --- state ---
        self.param_inputs = {}              # entries visibles en UI
        self.result_texts = {}
        self.model_metrics = {}             # métricas para export y FBN
        self.normal_fit_params = {}         # {model_name: {param: val}} al ejecutar fit normal
        self.bayes_fit_params = {}          # {model_name: {param: val}} al ejecutar BO
        self.last_fit_bayes = False
        self.current_canvas = None          # lienzo de la figura para poder destruir y re-pintar

        # --- modelos ---
        self.registered_models = {
            'Debye': DebyeModel(),
            'Cole-Cole': ColeColeModel(),
            'Cole-Davidson': ColeDavidsonModel(),
            'Havriliak-Negami': HavriliakNegamiModel(),
            'Fracc-1CR': Fractional1CRModel(),
            'Fracc-2CR': Fractional2CRModel(),
            'DebyeModulus': DebyeModulusModel(),
            'ColeColeModulus': ColeColeModulusModel(),
            'ColeDavidsonModulus': ColeDavidsonModulusModel(),
            'HNModulus': HNModulusModel(),
            'Fract1crModulus': Fract1crModulusModel(),
            'Fract2crModulus': Fract2crModulusModel(),
        }
        self.permittivity_models = {k:self.registered_models[k] for k in list(self.registered_models.keys())[:6]}
        self.modulus_models      = {k:self.registered_models[k] for k in list(self.registered_models.keys())[6:]}

        # --- layout ---
        self.top_control_frame = tk.Frame(root)
        self.top_control_frame.pack(fill=tk.X, padx=10, pady=5)

        self.load_button = tk.Button(self.top_control_frame, text="Load CSV", command=self.load_csv)
        self.load_button.grid(row=0, column=0, padx=5)

        self.domain_var = tk.StringVar(value="Permittivity")
        self.domain_combo = ttk.Combobox(self.top_control_frame, textvariable=self.domain_var, state="readonly")
        self.domain_combo['values'] = ("Permittivity", "Modulus")
        self.domain_combo.grid(row=0, column=1, padx=10)
        self.domain_combo.bind("<<ComboboxSelected>>", self.update_model_list)

        self.model_frame = tk.LabelFrame(self.top_control_frame, text="Select Models to Fit")
        self.model_frame.grid(row=0, column=2, sticky='nsw', padx=5)
        self.model_checks = {}

        self.params_container = tk.LabelFrame(self.top_control_frame, text="Model Parameters")
        self.params_container.grid(row=0, column=3, sticky='nsew', padx=5)
        self.params_canvas = tk.Canvas(self.params_container, height=220)
        self.params_scrollbar = tk.Scrollbar(self.params_container, orient="vertical", command=self.params_canvas.yview)
        self.params_frame = tk.Frame(self.params_canvas)
        self.params_frame.bind("<Configure>", lambda e: self.params_canvas.configure(scrollregion=self.params_canvas.bbox("all")))
        self.params_canvas.create_window((0,0), window=self.params_frame, anchor="nw")
        self.params_canvas.configure(yscrollcommand=self.params_scrollbar.set)
        self.params_canvas.pack(side="left", fill="both", expand=True)
        self.params_scrollbar.pack(side="right", fill="y")

        self.fit_button = tk.Button(self.top_control_frame, text="Fit Selected Models", command=self.run_fit)
        self.fit_button.grid(row=1, column=0, pady=(5,0), sticky='w')

        self.fit_bayes_button = tk.Button(self.top_control_frame, text="Bayesian Optimize + Fit", command=self.run_fit_bayes)
        self.fit_bayes_button.grid(row=1, column=1, pady=(5,0), sticky='w')

        self.reset_button = tk.Button(self.top_control_frame, text="Reset to Normal Fit", command=self.reset_to_normal)
        self.reset_button.grid(row=1, column=3, pady=(5,0), sticky='w')

        self.export_button = tk.Button(self.top_control_frame, text="Export Results", command=self.export_results)
        self.export_button.grid(row=1, column=4, pady=(5,0), sticky='w')

        self.fbn_button = tk.Button(self.top_control_frame, text="Evaluate Models with FBN", command=self.run_fbn)
        self.fbn_button.grid(row=1, column=2, pady=(5,0), sticky='w')

        # === Fit Results -> ahora ScrolledText (en lugar de Canvas+Frame) ===
        self.result_container = tk.LabelFrame(self.top_control_frame, text="Fit Results")
        self.result_container.grid(row=0, column=5, rowspan=2, sticky='nsew', padx=5)
        self.result_text = scrolledtext.ScrolledText(self.result_container, height=14, width=70, wrap=tk.WORD)
        self.result_text.pack(fill='both', expand=True)

        self.plot_frame = tk.Frame(root, height=520)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)

        self.update_model_list()

    # ---------- utils ----------
    def _clear_plots_and_results(self):
        for w in self.plot_frame.winfo_children():
            w.destroy()
        # limpiamos el ScrolledText (ya no destruimos widgets porque sólo hay uno)
        self.result_text.delete("1.0", tk.END)
        self.current_canvas = None

    def _selected_models(self):
        return [name for name, var in self.model_checks.items() if var.get()]

    def _inputs_to_param_dict(self, model_name):
        d = {}
        for key, cells in self.param_inputs[model_name].items():
            try:
                v  = float(cells['val'].get())
                vmin = float(cells['min'].get())
                vmax = float(cells['max'].get())
            except Exception:
                v, vmin, vmax = 0.0, -np.inf, np.inf
            d[key] = {'val': v, 'min': vmin, 'max': vmax}
        return d

    def _apply_params_to_entries(self, model_name, param_map):
        # Escribe valores en los entries visibles
        for k, v in param_map.items():
            if model_name in self.param_inputs and k in self.param_inputs[model_name]:
                e = self.param_inputs[model_name][k]['val']
                e.delete(0, tk.END)
                e.insert(0, f"{float(v):.6g}")

    def _safe_fit(self, model_instance, freq, eps_real, eps_imag, param_dict):
        """Llama model.fit protegido con try/except; retorna (resR, resI) o (None, None)."""
        try:
            # Garantiza que los parámetros iniciales del modelo se actualicen
            if hasattr(model_instance, 'params'):
                for k, spec in param_dict.items():
                    try:
                        model_instance.params[k].set(value=spec['val'], min=spec['min'], max=spec['max'])
                    except Exception:
                        pass
            return model_instance.fit(freq, eps_real, eps_imag, param_dict)
        except Exception as e:
            print(f"Fit error in {type(model_instance).__name__}: {e}")
            return None, None

    # ---------- update model list ----------
    def update_model_list(self, event=None):
        for w in self.model_frame.winfo_children():
            w.destroy()
        self.model_checks.clear()
        models_to_show = self.permittivity_models if self.domain_var.get() == "Permittivity" else self.modulus_models
        for model_name in models_to_show:
            var = tk.BooleanVar()
            chk = tk.Checkbutton(self.model_frame, text=model_name, variable=var, command=self.update_params)
            chk.pack(anchor='w')
            self.model_checks[model_name] = var
        # refrescar el panel de parámetros
        self.update_params()

    # ---------- update params (panel) ----------
    def update_params(self):
        for w in self.params_frame.winfo_children():
            w.destroy()
        self.param_inputs.clear()
        # Crear entradas solo para modelos seleccionados
        for model_name, var in self.model_checks.items():
            if var.get():
                model_instance = self.registered_models[model_name]
                params = model_instance.get_params()
                tk.Label(self.params_frame, text=f"--- {model_name} Parameters ---", font=('Arial', 10, 'bold')).pack(anchor='w', pady=(6,2))
                self.param_inputs[model_name] = {}
                for key, (val, vmin, vmax) in params.items():
                    f = tk.Frame(self.params_frame)
                    f.pack(anchor='w')
                    tk.Label(f, text=f"{key}:").grid(row=0, column=0, padx=(0,4))
                    e_val = tk.Entry(f, width=10)
                    e_val.insert(0, f"{0.0 if val is None else float(val):.6g}")
                    e_val.grid(row=0, column=1)
                    e_min = tk.Entry(f, width=10)
                    e_min.insert(0, str(vmin))
                    e_min.grid(row=0, column=2)
                    e_max = tk.Entry(f, width=10)
                    e_max.insert(0, str(vmax))
                    e_max.grid(row=0, column=3)
                    self.param_inputs[model_name][key] = {'val': e_val, 'min': e_min, 'max': e_max}

    # ---------- load CSV ----------
    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files","*.csv")])
        if not file_path:
            return
        df = pd.read_csv(file_path, encoding="utf-8-sig")
        df.columns = df.columns.str.strip()
        try:
            self.freq = df['frequency'].values
            real = df['eps_real'].values
            imag = df['eps_imag'].values
            self.data = real + 1j*imag
            messagebox.showinfo("Success","CSV loaded successfully.")
            # autoinit params (si el modelo lo soporta)
            for _name, m in self.registered_models.items():
                if hasattr(m, 'set_auto_params_from_data'):
                    try:
                        m.set_auto_params_from_data(np.real(self.data))
                    except Exception:
                        pass
            self.update_params()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {e}")

    # ---------- public runners ----------
    def run_fit(self):
        self._fit_core(use_bayes=False)

    def run_fit_bayes(self):
        self._fit_core(use_bayes=True)

    # ---------- core fit (normal y BO) ----------
    def _fit_core(self, use_bayes=False):
        if self.data is None or self.freq is None:
            messagebox.showwarning("Missing Data","Please load data first.")
            return

        self._clear_plots_and_results()

        fig, axs = plt.subplots(2, 2, figsize=(10, 8), dpi=100)
        axs = axs.flatten()
        eps_real = np.real(self.data)
        eps_imag = np.imag(self.data)
        tan_delta = eps_imag / np.where(eps_real==0, np.nan, eps_real)

        axs[0].set_title("Real part ε′")
        axs[1].set_title("Imaginary part ε″")
        axs[2].set_title("Tan δ = ε″/ε′")
        axs[3].set_title("Cole-Cole plot: ε″ vs ε′")
        axs[0].semilogx(self.freq, eps_real, 'k.', label='Data')
        axs[1].semilogx(self.freq, eps_imag, 'k.', label='Data')
        axs[2].semilogx(self.freq, tan_delta, 'k.', label='Data')
        axs[3].plot(eps_real, eps_imag, 'ko', label='Data')

        self.model_metrics.clear()
        if not use_bayes:
            # Sólo sobreescribimos normal_fit_params cuando el usuario pide un fit normal nuevo
            self.normal_fit_params.clear()
        else:
            self.bayes_fit_params.clear()

        selected = self._selected_models()
        if not selected:
            messagebox.showwarning("No models","Please select at least one model.")
            return

        for model_name in selected:
            model_instance = self.registered_models[model_name]

            # Lee parámetros actuales de la UI
            param_dict = self._inputs_to_param_dict(model_name)

            # ============= Bayesian Optimize (opcional) =============
            if use_bayes:
                # Si existe fit normal previo para este modelo, lo usamos como base
                if model_name in self.normal_fit_params and self.normal_fit_params[model_name]:
                    base = self.normal_fit_params[model_name]
                else:
                    # si no, usamos lo que esté en los entries como base
                    base = {k: spec['val'] for k, spec in param_dict.items()}

                # Definimos un espacio de búsqueda acotado en torno al fit normal
                search_space = {}
                for k, spec in param_dict.items():
                    v0 = float(base.get(k, spec['val']))
                    # ancho relativo 50% alrededor del valor base, respetando min/max absolutos
                    width = max(abs(v0) * 0.5, 1e-12)
                    lo = max(float(spec['min']), v0 - width)
                    hi = min(float(spec['max']), v0 + width)
                    # si min>=max por cualquier razón, abrimos al rango original
                    if lo >= hi:
                        lo, hi = float(spec['min']), float(spec['max'])
                    search_space[k] = (lo, hi)

                def objective(params_dict):
                    # construimos un param_dict válido para iniciar el fit determinista
                    trial = {k: {'val': float(v), 'min': search_space[k][0], 'max': search_space[k][1]}
                             for k, v in params_dict.items()}
                    resR, resI = self._safe_fit(model_instance, self.freq, eps_real, eps_imag, trial)
                    if resR is None or resI is None:
                        return 1e9  # gran penalización si el fit falla
                    mse_total = mean_squared_error(eps_real, resR.best_fit) + \
                                mean_squared_error(eps_imag, resI.best_fit)
                    return mse_total  # minimizamos MSE total

                bo = BayesianStarter(objective, search_space)
                try:
                    best_params = bo.run(n_iter=30)  # asumiendo API (minimiza objective)
                except Exception as e:
                    print(f"BO failed for {model_name}: {e}")
                    best_params = base  # fallback

                # volcamos mejores parámetros al UI y al param_dict
                self._apply_params_to_entries(model_name, best_params)
                for k in param_dict:
                    if k in best_params:
                        param_dict[k]['val'] = float(best_params[k])

            # ============= Fit determinista (normal o refinamiento tras BO) =============
            res_real, res_imag = self._safe_fit(model_instance, self.freq, eps_real, eps_imag, param_dict)
            if res_real is None or res_imag is None:
                # mostramos aviso en el panel de resultados y continuamos
                self.result_text.insert(tk.END, f"===== {model_name} =====\nFit failed. Check bounds/initial values.\n\n")
                self.result_text.see(tk.END)
                continue

            eps_r_fit = res_real.best_fit
            eps_i_fit = res_imag.best_fit
            with np.errstate(divide='ignore', invalid='ignore'):
                tan_d_fit = eps_i_fit / eps_r_fit

            label = model_name + (" (BO)" if use_bayes else "")
            axs[0].semilogx(self.freq, eps_r_fit, label=label)
            axs[1].semilogx(self.freq, eps_i_fit, label=label)
            axs[2].semilogx(self.freq, tan_d_fit, label=label)
            axs[3].plot(eps_r_fit, eps_i_fit, label=label)

            # guardado de parámetros usados para reproducibilidad
            fitted_params = {pname: p.value for pname, p in res_imag.params.items()} if hasattr(res_imag, 'params') else {k: v['val'] for k, v in param_dict.items()}

            if use_bayes:
                self.bayes_fit_params[model_name] = dict(fitted_params)
            else:
                self.normal_fit_params[model_name] = dict(fitted_params)

            # ----- resultados y métricas (en ScrolledText) -----
            self.result_text.insert(tk.END, f"===== {model_name}{' (BO)' if use_bayes else ''} =====\n\n")
            for pname, pval in fitted_params.items():
                self.result_text.insert(tk.END, f"{pname} = {float(pval):.6g}\n")

            r2_real, mse_real, aad_real = calculate_fit_metrics(eps_real, eps_r_fit)
            r2_imag, mse_imag, aad_imag = calculate_fit_metrics(eps_imag, eps_i_fit)
            self.result_text.insert(tk.END, f"\n--- Real part metrics ---\nR²={r2_real:.4f}, MSE={mse_real:.4g}, AAD={aad_real:.4g}\n")
            self.result_text.insert(tk.END, f"\n--- Imag part metrics ---\nR²={r2_imag:.4f}, MSE={mse_imag:.4g}, AAD={aad_imag:.4g}\n\n")
            self.result_text.see(tk.END)

            self.model_metrics[model_name] = {
                'r2_real': r2_real, 'mse_real': mse_real, 'aad_real': aad_real,
                'r2_imag': r2_imag, 'mse_imag': mse_imag, 'aad_imag': aad_imag,
            }

        for ax in axs:
            ax.legend(); ax.grid(True)
        fig.tight_layout()
        self.current_canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.current_canvas.draw()
        self.current_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.last_fit_bayes = use_bayes

    # ---------- reset ----------
    def reset_to_normal(self):
        if not self.normal_fit_params:
            messagebox.showwarning("No normal fit stored","Run a normal fit first.")
            return
        # Restaura los parámetros en UI para los modelos seleccionados (si tenemos snapshot)
        restored_any = False
        for model_name in self._selected_models():
            if model_name in self.normal_fit_params:
                self._apply_params_to_entries(model_name, self.normal_fit_params[model_name])
                restored_any = True
        if not restored_any:
            messagebox.showinfo("Reset","No selected model has a stored normal fit yet.")
            return
        self.last_fit_bayes = False
        # Re-pinta inmediatamente con el fit normal
        self.run_fit()

    # ---------- export ----------
    def export_results(self):
        if not self.model_metrics:
            messagebox.showwarning("No Data","Run fitting first!")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files","*.csv")])
        if not file_path:
            return
        rows = []
        for model_name, metrics in self.model_metrics.items():
            row = {'Model': model_name}
            # Exporta los parámetros actualmente visibles en UI (lo que el usuario ve)
            if model_name in self.param_inputs:
                for key in self.param_inputs[model_name]:
                    try:
                        row[key] = float(self.param_inputs[model_name][key]['val'].get())
                    except Exception:
                        row[key] = None
            row.update(metrics)
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(file_path, index=False)
        messagebox.showinfo("Export","Results exported successfully.")

    # ---------- FBN ----------
    def run_fbn(self):
        if not self.model_metrics:
            messagebox.showwarning("No Metrics","Run fitting first!")
            return
        scores = {}
        for model, m in self.model_metrics.items():
            score_real = m['r2_real'] / (m['mse_real'] + m['aad_real'] + 1e-6)
            score_imag = m['r2_imag'] / (m['mse_imag'] + m['aad_imag'] + 1e-6)
            scores[model] = 0.5 * (score_real + score_imag)
        total = sum(max(s, 0) for s in scores.values())
        if total <= 0:
            messagebox.showinfo("FBN Analysis", "Scores are non-positive; cannot compute probabilities.")
            return
        probs = {k: max(v, 0)/total for k, v in scores.items()}
        sorted_probs = dict(sorted(probs.items(), key=lambda x: x[1], reverse=True))
        msg = "FBN Model Probabilities:\n"
        for k, v in sorted_probs.items():
            msg += f"{k}: {v*100:.1f}%\n"
        messagebox.showinfo("FBN Analysis", msg)

# -------------------- main --------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = DielectricGUI(root)
    root.mainloop()
