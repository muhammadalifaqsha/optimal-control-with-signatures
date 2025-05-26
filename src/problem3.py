from random import random
import numpy as np
from scipy.interpolate import CubicSpline
from tqdm import tqdm
from scipy.integrate import solve_ivp
import tensor_algebra as ta
import iisignature

class BrownianSignatureFunctional:
    """
    Model parameters for the environment.
    """
    def __init__(self, dim, order, T, Nt):
        self.T = T
        self.Nt = Nt
        self.timesteps = np.linspace(0, self.T, num = (Nt+1))
        self.order = order
        self.dim = dim
        
    def simulate_BM_sig(self, nsims=1):
        W = np.zeros((self.Nt+1, nsims))
        W_sig_progression = []
        dt = self.T/(self.Nt)
        errs = np.random.randn(self.Nt, nsims)
        order = self.order
        for t in range(self.Nt):
            W[t + 1,:] = W[t,:] + np.sqrt(dt) * errs[t,:]
            W_stopped = W[:(t+1),:]
            timesteps_stopped = self.timesteps[:(t+1)]
            if t == 0:
                W_sig = ta.one(dim = 2)
                W_sig_stopped = [W_sig for j in range(nsims)]
            else:
                W_sig_stopped = []
                for i in range(nsims):
                    path = np.array([timesteps_stopped,W_stopped[:, i]]).T
                    W_sig_flat = iisignature.sig(path, order)
                    W_sig_flat = np.insert(W_sig_flat, 0, 1)
                    W_sig = ta.sig_to_tensor(W_sig_flat, 2, order)
                    W_sig_stopped.append(W_sig)
            W_sig_progression.append(W_sig_stopped)                    
        return W, W_sig_progression