### models/Ionic.py

from .base_model import BaseModel
import numpy as np
from lmfit import Model, Parameters

class IonicModel(BaseModel):
    def __init__(self):
        super().__init__("Ionic")
        self.params_init = {
            'eps_inf1': (None, 0.5, 10),
            'eps_s1': (None, 0.6, 500),
            'eps_inf2': (None, 0.5, 10),
            'eps_s2': (None, 0.6, 500),
            'tau_alpha': (1e-4, 1e-9, 1e-1),
            'tau_beta': (1e-4, 1e-9, 1e-1),
            'tau_gamma': (1e-4, 1e-9, 1e-1),
            'alpha': (0.5, 0.01, 1.0), 
            'beta': (0.5, 0.01, 1.0),   
            'gamma': (0.5, 0.01, 1.0),
        }

    def get_params(self):
        return self.params_init

    def set_auto_params_from_data(self, eps_real, n_points=5):
        flex_factor = 1.5  # Ampliar el rango hasta ±150%

        avg_low_freq = np.mean(eps_real[:n_points])
        avg_high_freq = np.mean(eps_real[-n_points:])

        _, min_s, max_s = self.params_init['eps_s1']
        _, min_inf, max_inf = self.params_init['eps_inf1']
        _, min_s, max_s = self.params_init['eps_s2']
        _, min_inf, max_inf = self.params_init['eps_inf2']

        self.params_init['eps_s1'] = (
            avg_low_freq,
            min_s,
            max(max_s, avg_low_freq * flex_factor)
        )
        self.params_init['eps_inf1'] = (
            avg_high_freq,
            min_inf,
            max(max_inf, avg_high_freq * flex_factor)
        )    
        self.params_init['eps_s2'] = (
            avg_low_freq,
            min_inf,
            max(max_inf, avg_high_freq * flex_factor)
        )    
        self.params_init['eps_inf2'] = (
            avg_high_freq,
            min_inf,
            max(max_inf, avg_high_freq * flex_factor)
        )    

    def model_function(self, f, eps_inf1, eps_s1, eps_inf2, eps_s2, tau_alpha, tau_beta, tau_gamma, alpha, beta, gamma):
        w = 2 * np.pi * f

        A1 = (w * tau_alpha) ** (-alpha) * np.cos(alpha * np.pi / 2) + \
             (w * tau_beta) ** (-beta) * np.cos(beta * np.pi / 2)
        A2 = (w * tau_alpha) ** (-alpha) * np.sin(alpha * np.pi / 2) + \
             (w * tau_beta) ** (-beta) * np.sin(beta * np.pi / 2)
        A3 = (w * tau_gamma) ** (-gamma) * np.cos(gamma * np.pi / 2)
        A4 = (w * tau_gamma) ** (-gamma) * np.sin(gamma * np.pi / 2)

        # Ecuaciones de ε′ y ε″ como parte de un número complejo
        eps_real = (eps_s1 - ((eps_s1 - eps_inf1) * (1 + A1)) / ((1 + A1)**2 + A2**2))+(eps_s2-eps_inf2)*A3
        eps_imag = (((eps_s1 - eps_inf1) * A2) / ((1 + A1)**2 + A2**2))+(eps_s2-eps_inf2)*A4

        return eps_real + 1j * eps_imag  # NOTA: Usamos +1j aquí porque lmfit maneja np.real e np.imag por separado

    def fit(self, f, eps_real, eps_imag, user_params=None):
        def model_real(f, eps_inf1, eps_s1, eps_inf2, eps_s2, tau_alpha, tau_beta, tau_gamma, alpha, beta, gamma):
            return np.real(self.model_function(f, eps_inf1, eps_s1, eps_inf2, eps_s2, tau_alpha, tau_beta, tau_gamma, alpha, beta, gamma))

        def model_imag(f, eps_inf1, eps_s1, eps_inf2, eps_s2, tau_alpha, tau_beta, tau_gamma, alpha, beta, gamma):
            return np.imag(self.model_function(f, eps_inf1, eps_s1, eps_inf2, eps_s2, tau_alpha, tau_beta, tau_gamma, alpha, beta, gamma))

        model_real_fit = Model(model_real)
        model_imag_fit = Model(model_imag)

        params = Parameters()

        if user_params:
            for key in self.params_init:
                _, minval, maxval = self.params_init[key]
                val_str = user_params[key]['val']
                val = float(val_str) if val_str not in ('', None) else None

                if val is None:
                    val = self.params_init[key][0]

                params.add(key, value=val, min=minval, max=maxval)
        else:
            for key, (val, minval, maxval) in self.params_init.items():
                if val is None:
                    raise ValueError(f"Missing automatic value for parameter '{key}'")
                params.add(key, value=val, min=minval, max=maxval)

        result_real = model_real_fit.fit(eps_real, f=f, params=params)
        result_imag = model_imag_fit.fit(eps_imag, f=f, params=result_real.params)

        self.params = result_imag.params
        return result_real, result_imag
