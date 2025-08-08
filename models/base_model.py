# models/base_model.py

from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, name):
        self.name = name
        self.params = {}

    @abstractmethod
    def model_function(self, f, **params):
        pass

    def initial_guess(self, f, eps_real, eps_imag):
        raise NotImplementedError("Initial guess must be implemented in subclass.")

    def fit(self, f, eps_real, eps_imag):
        raise NotImplementedError("Fit logic must be implemented in subclass.")
