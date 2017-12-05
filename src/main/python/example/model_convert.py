import numpy as np

from ndnn.store import ParamStore

store = ParamStore('s2s_lstm.mdl')

params = store.load()

name = ['enc_embed', 'enc_wf', 'enc_bf', 'enc_wi', 'enc_bi',
        'enc_wc', 'enc_bc', 'enc_wo', 'enc_bo', 'enc_v2c',
        'dec_embed', 'dec_wf', 'dec_bf', 'dec_wi', 'dec_bi',
        'dec_wc', 'dec_bc', 'dec_wo', 'dec_bo', 'dec_v2c']

for i, n in enumerate(name):
    np.savetxt(n + ".npy", params[i], delimiter='\t')

"""
self.C2V = self.param_of((num_char, hidden_dim))
self.wf = self.param_of((2 * hidden_dim, hidden_dim))
self.bf = self.param_of((hidden_dim), Zero())
self.wi = self.param_of((2 * hidden_dim, hidden_dim))
self.bi = self.param_of((hidden_dim), Zero())
self.wc = self.param_of((2 * hidden_dim, hidden_dim))
self.bc = self.param_of((hidden_dim), Zero())
self.wo = self.param_of((2 * hidden_dim, hidden_dim))
self.bo = self.param_of((hidden_dim), Zero())
self.V = self.param_of((hidden_dim, num_char))
"""
