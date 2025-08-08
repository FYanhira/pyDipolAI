### models/coledavidson.py

from .base_model import BaseModel
import numpy as np
from lmfit import Model, Parameters

class ColeDavidsonModel(BaseModel):
    def __init__(self):
        super().__init__("Cole-Davidson")
        self.params_init = {
            'eps_inf': (None, 0.5, 10),
            'eps_s': (None, 0.6, 20),
            'tau': (1e-4, 1e-9, 1e-1),
            'beta': (0.2, 0.0, 1.0),
        }

    def get_params(self):
        return self.params_init

    def set_auto_params_from_data(self, eps_real, n_points=5):
        avg_low_freq = np.mean(eps_real[:n_points])
        avg_high_freq = np.mean(eps_real[-n_points:])

        _, min_s, max_s = self.params_init['eps_s']
        _, min_inf, max_inf = self.params_init['eps_inf']

        self.params_init['eps_s'] = (avg_low_freq, min_s, max_s)
        self.params_init['eps_inf'] = (avg_high_freq, min_inf, max_inf)

    def model_function(self, f, eps_inf, eps_s, tau, beta):
            w = 2 * np.pi * f
            delta_eps = eps_s - eps_inf
            wtau = w * tau

            mag = (1 + wtau**2) ** (-beta / 2)
            theta = np.arctan(wtau)

            eps_real = eps_s - delta_eps * mag * np.cos(beta * theta)
            eps_imag = delta_eps * mag * np.sin(beta * theta)

            return eps_real + 1j * eps_imag  # También válido: eps_real + 1j * (-eps_imag)

    def fit(self, f, eps_real, eps_imag, user_params=None):
        def model_real(f, eps_inf, eps_s, tau, beta):
            return np.real(self.model_function(f, eps_inf, eps_s, tau, beta))

        def model_imag(f, eps_inf, eps_s, tau, beta):
            return np.imag(self.model_function(f, eps_inf, eps_s, tau, beta))

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
