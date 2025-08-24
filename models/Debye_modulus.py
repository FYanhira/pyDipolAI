from .base_model import BaseModel
import numpy as np
from lmfit import Model, Parameters

class DebyeModulusModel(BaseModel):
    def __init__(self):
        super().__init__("Debye Modulus")
        # Inicialización: tau fijo, M_s y M_inf se calculan de datos
        self.params_init = {
            'M_inf': (None, None, None),  # se definen en set_auto_params_from_data
            'M_s': (None, None, None),    # se definen en set_auto_params_from_data
            'tau': (1e-4, 1e-6, 1e-1),    # valor inicial y rango fijos
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
        self.params_init['M_s'] = (
            avg_low_freq,
            avg_low_freq / flex_factor,
            avg_low_freq * flex_factor
        )
        self.params_init['M_inf'] = (
            avg_high_freq,
            avg_high_freq / flex_factor,
            avg_high_freq * flex_factor
        )

    def model_function(self, f, M_inf, M_s, tau):
        w = 2 * np.pi * f
        denom = M_inf**2 + (M_s * w * tau)**2
    
        M_real = (M_inf * M_s) * (M_inf + M_s * (w * tau)**2) / denom
        M_imag = (M_inf * M_s) * (M_inf - M_s) * (w * tau) / denom
        return M_real + 1j * M_imag

    def fit(self, f, M_real, M_imag, user_params=None):
        def model_real(f, M_inf, M_s, tau):
            return np.real(self.model_function(f, M_inf, M_s, tau))

        def model_imag(f, M_inf, M_s, tau):
            return np.imag(self.model_function(f, M_inf, M_s, tau))

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
