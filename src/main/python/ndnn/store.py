import os
import pickle


class ParamStore:
    def __init__(self, model_file):
        self.model = model_file
    
    def load(self):
        if os.path.exists(self.model):
            with open(self.model, 'rb') as f:
                p_value = pickle.load(f)
                param_store = []
                for p in p_value:
                    param_store.append(p)
                return param_store
        return None

    def store(self, params):
        f = open(self.model, 'wb')
        pickle.dump(params, f)
