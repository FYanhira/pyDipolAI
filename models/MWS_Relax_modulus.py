from .base_model import BaseModel
import numpy as np
from lmfit import Model, Parameters

class MWSModulusModel(BaseModel):
    def __init__(self):
        super().__init__("MWS Modulus")
        # Inicialización: tau fijo, M_s y M_inf se calculan de datos
        self.params_init = {
            'M_inf1': (None, None, None),  # se definen en set_auto_params_from_data
            'M_s1': (None, None, None),    # se definen en set_auto_params_from_data
            'M_inf2': (None, None, None),  
            'M_s2': (None, None, None), 
            'tau_alpha': (1e-4, 1e-6, 1e-1),    # valor inicial y rango fijos
            'tau_beta': (1e-4, 1e-6, 1e-1),
            'tau_gamma': (1e-4, 1e-6, 1e-1),
            'alpha': (0.5, 0.0, 1.0),
            'beta': (0.5, 0.0, 1.0),
            'gamma': (0.5, 0.0, 1.0),
        }

    def get_params(self):
        return self.params_init

    def set_auto_params_from_data(self, M_real, n_points=5):
        """
        Calcula M_s y M_inf automáticos y sus rangos en base a datos.
        """
        flex_factor = 2

        avg_low_freq = np.mean(M_real[:n_points])   # baja frecuencia
        avg_high_freq = np.mean(M_real[-n_points:]) # alta frecuencia

        # Definir valores y rangos calculados dinámicamente
        self.params_init['M_s1'] = (
            avg_low_freq,
            avg_low_freq / flex_factor,
            avg_low_freq * flex_factor
        )
        self.params_init['M_inf1'] = (
            avg_high_freq,
            avg_high_freq / flex_factor,
            avg_high_freq * flex_factor
        )
        self.params_init['M_s2'] = (
            avg_low_freq,
            avg_low_freq / flex_factor,
            avg_low_freq * flex_factor
        )
        self.params_init['M_inf2'] = (
            avg_high_freq,
            avg_high_freq / flex_factor,
            avg_high_freq * flex_factor
        )

    def model_function(self, f, M_inf1, M_s1, M_inf2, M_s2, tau_alpha, tau_beta, tau_gamma, alpha, beta, gamma):
        w = 2 * np.pi * f
        
        denom = (M_inf1**2 + 2 * M_s1 * M_inf1 * (np.cos(gamma * np.pi / 2) * (w * tau_gamma)**gamma) + (M_s1 * (w * tau_gamma)**gamma)**2)
        M_real1 = (M_inf1 * M_s1 * (M_inf1 + (M_s1 + M_inf1) * (np.cos(gamma * np.pi / 2) * (w * tau_gamma)**gamma) + M_s1 * (w * tau_gamma)**(2 * gamma))) / denom
        M_imag1 = (M_inf1 * M_s1 * (M_inf1 - M_s1) * (np.sin(gamma * np.pi / 2) * (w * tau_gamma)**gamma)) / denom
        
        A1 = (np.cos(alpha * np.pi / 2) * (w * tau_alpha)**(-alpha) + np.cos(beta * np.pi / 2) * (w * tau_beta)**(-beta))
        A2 = (np.sin(alpha * np.pi / 2) * (w * tau_alpha)**(-alpha) + np.sin(beta * np.pi / 2) * (w * tau_beta)**(-beta))
        M_real2 = (M_inf2 * M_s2 * (M_s2 + A1 * (M_s2 + M_inf2) + M_inf2 * (A1**2 + A2**2))) / ((M_s2 + M_inf2 * A1)**2 + (M_inf2 * A2)**2)
        M_imag2 = (M_inf2 * M_s2 * (M_inf2 - M_s2) * A2) / ((M_s2 + M_inf2 * A1)**2 + (M_inf2 * A2)**2)

        M_real = M_real1 + M_real2
        M_imag = M_imag1 + M_imag2
        return M_real + 1j * M_imag

    def fit(self, f, M_real, M_imag, user_params=None):
        def model_real(f, M_inf1, M_s1, M_inf2, M_s2, tau_alpha, tau_beta, tau_gamma, alpha, beta, gamma):
            return np.real(self.model_function(f, M_inf1, M_s1, M_inf2, M_s2, tau_alpha, tau_beta, tau_gamma, alpha, beta, gamma))

        def model_imag(f, M_inf1, M_s1, M_inf2, M_s2, tau_alpha, tau_beta, tau_gamma, alpha, beta, gamma):
            return np.imag(self.model_function(f, M_inf1, M_s1, M_inf2, M_s2, tau_alpha, tau_beta, tau_gamma, alpha, beta, gamma))

        model_real_fit = Model(model_real)
        model_imag_fit = Model(model_imag)

        params = Parameters()

        if user_params:
            # Valores desde GUI o entrada del usuario
            for key in self.params_init:
                _, minval, maxval = self.params_init[key]
                val_str = user_params[key]['val']
                val = float(val_str) if val_str not in ('', None) else None

                if val is None:
                    val = self.params_init[key][0]

                params.add(key, value=val, min=minval, max=maxval)
        else:
            # Valores automáticos desde set_auto_params_from_data
            for key, (val, minval, maxval) in self.params_init.items():
                if val is None or minval is None or maxval is None:
                    raise ValueError(f"Missing automatic value for parameter '{key}'. Did you call set_auto_params_from_data?")
                params.add(key, value=val, min=minval, max=maxval)

        result_real = model_real_fit.fit(M_real, f=f, params=params)
        result_imag = model_imag_fit.fit(M_imag, f=f, params=result_real.params)

        self.params = result_imag.params
        return result_real, result_imag
