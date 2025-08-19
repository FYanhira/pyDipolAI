class BayesianStarter:
    def __init__(self, objective_func, search_space):
        self.objective_func = objective_func
        self.search_space = search_space

    def run(self, n_iter=20):
        # Aquí pondrías tu código real de optimización
        # Por ahora devolvemos parámetros de prueba
        best_params = {k: (v[0] + v[1])/2 for k,v in self.search_space.items()}
        return best_params
