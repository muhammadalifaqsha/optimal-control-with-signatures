import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from math import factorial
import os
import datetime
from time import time
from scipy import optimize

import torch
from torch import optim
torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)

import iisignature
from problem2 import *

import tensor_algebra as ta

class Optimiser:
    """Solves the optimisation problem on the tensor algebra."""

    def __init__(self, env, W_sig, order = 3, poly_deg = 3):
        self.env = env
        self.dim = self.env.dim
        self.order = order
        self.history = []
        self.e0 = ta.word((), self.dim)
        self.e1 = ta.word((0,), self.dim)
        self.e2 = ta.word((1,), self.dim)
        self.poly_deg = poly_deg
        self.Q0 = self.e0*self.env.Q0

        # Define the linear functionals we'll be solving for.
        self.F = ta.Tensor(env.dim, self.order)
        self.F.value = [torch.randn([env.dim] * n, requires_grad=True) for n in range(self.order + 1)]
        #self.F.value = [torch.zeros([env.dim] * n, requires_grad=True) for n in range(self.order + 1)]
 
        # Define the expected signatures of Brownian motion
        T = self.env.T
        #expected_sig = [1, T, 0, T**2/2,  0, 0, T/2, T**3/6, 0, 0, T**2/4, 0, 0, T**2/4, 0]
        #self.W_sig_expected = ta.sig_to_tensor(expected_sig, dim = self.dim, order = self.order)
        self.W_sig_expected = W_sig.truncate(self.order)

    @staticmethod
    def _sample(path, points=100):
        times = np.linspace(path[0, 0], path[-1, 0], points)
        idx = np.searchsorted(path[:, 0], times, side="left")
        #idx = np.unique(np.linspace(0, len(path) - 1, points).astype(int))


        return path[idx]

    def value_fn(self, F, trunc_lev_s = 3, MAX_TAYLOR = 100):
        #F_dim = len(F.value)
        dim = self.dim
        #F = ta.sig_to_tensor(F.value, self.env.dim, self.order)
        
        # Define tensor algebra constant
        #e0 = ta.word((), dim)
        #e1 = ta.word((0,), dim)
        #e2 = ta.word((1,), dim)
        
        # Defining the representation of S and \hat{\alpha} in terms of the innovation process
        trunc_lev = trunc_lev_s
        
        LG = self.e1*self.env.b2 + F*self.env.b3*self.e1 + self.e2*self.env.sigma2
        neg_LG = -LG
        
        max_iter = F.order #200
        M_max = F.order
        
        exp_LG = LG.exp(MAX_ITER = max_iter, MAX_LEVEL = trunc_lev, METHOD = '2')
        exp_neg_LG = neg_LG.exp(MAX_ITER = max_iter, MAX_LEVEL = trunc_lev, METHOD = '2')
        
        LF = exp_LG*self.env.S0
        term1 = (exp_neg_LG.shuffle(self.e0*self.env.b0 + F*self.env.b1, max_order = M_max))*self.e1
        LF += exp_LG.shuffle(term1, max_order = M_max)
        term2 = (exp_neg_LG*self.env.sigma0)*self.e2
        LF += exp_LG.shuffle(term1, max_order = M_max)
        
        LF = LF.truncate(trunc_lev)

        self.LF = LF
        LQ = self.e0*self.env.Q0 + F*self.e1
        LQ2 = self.e0*(self.env.Q0)**2 + 2*LQ.shuffle(F, max_order = M_max)*self.e1        

        # Define functional of performance criterion
        
        M_max = F.order
        #M_max = 2*F.order
        p_perf = -F.shuffle(F, max_order = M_max)
        p_perf = p_perf*(self.e1*self.env.k)
        #M_max = (F*self.e1).order + F.order
        p_perf += LQ*(self.e1*self.env.b0)
        p_perf += LQ.shuffle(F, max_order = M_max)*(self.e1*self.env.b1)
        p_perf += LQ.shuffle(LF, max_order = M_max)*(self.e1*self.env.b2)
        p_perf += (LQ.shuffle(LF, max_order = M_max)).shuffle(F, max_order = M_max)*(self.e1*self.env.b3)
        p_perf += LQ.shuffle(LF, max_order = M_max)*(self.e1*self.env.sigma2*0.5)
        #M_max = (F*self.e1).order + trunc_lev_1
        #p_perf += LQ*(self.e2*self.env.sigma0)
        #p_perf += LQ.shuffle(LF, max_order = M_max)*(self.e2*self.env.sigma2)
        #M_max = 2*F_prod_e1.order
        p_perf -= (LQ2)*(self.e1*self.env.theta2)
        p_perf -= (LQ2 - self.e0*self.env.Q0**2)*(self.env.theta1)
        p_perf = p_perf.truncate(self.order)
        #p_perf += (F_prod_e1)*(self.e2*self.env.sigma_s)
        
        #W_sig_expected = expected_signature_bm(self.env, order = self.order, nsims = 1_000)
        
        #criterion = p_perf.dot(W_sig_expected)
        criterion = -p_perf.dot(self.W_sig_expected)
        
        self.p_perf = p_perf

        return criterion
        
    def train_sgd(self, n_epochs, lr=0.01, trunc_lev_s = 3, method="SGD", MAX_TAYLOR=100):
        
        # Define Torch optimiser.
        if method == "SGD":
            optimiser = optim.SGD(self.F.value, lr=lr)
        elif method == "Adam":
            optimiser = optim.Adam(self.F.value, lr=lr)
        else:
            raise ValueError("Method {} not supported.".format(method))

        pbar = tqdm(range(n_epochs))

        for _ in pbar:
            optimiser.zero_grad()

            # Define polynomial on the dual of the tensor algebra

            # Defining the representation of S and \hat{\alpha} in terms of the innovation process

            #criterion = p_perf.dot(W_sig_expected)
            loss = self.value_fn(self.F, trunc_lev_s, MAX_TAYLOR)

            # Train
            loss.backward()
            optimiser.step()

            loss_val = -loss.detach().numpy()
            pbar.set_description(str(loss_val))
            self.history.append(loss_val)