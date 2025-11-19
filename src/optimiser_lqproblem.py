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

    def __init__(self, env, W_sig, order = 3, max_order_fn = 3):
        self.env = env
        self.dim = self.env.dim
        self.order = order
        self.max_order_fn = max_order_fn
        self.history = []
        self.e0 = ta.word((), self.dim)
        self.e1 = ta.word((0,), self.dim)
        self.e2 = ta.word((1,), self.dim)
        self.X0 = self.e0*self.env.X0

        # Define the linear functionals we'll be solving for.
        self.F = ta.Tensor(env.dim, self.order)
        self.F.value = [torch.randn([env.dim] * n, requires_grad=True) for n in range(self.order + 1)]
        #self.F.value = [torch.zeros([env.dim] * n, requires_grad=True) for n in range(self.order + 1)]
 
        # Define the expected signatures of Brownian motion
        T = self.env.T
        #expected_sig = [1, T, 0, T**2/2,  0, 0, T/2, T**3/6, 0, 0, T**2/4, 0, 0, T**2/4, 0]
        #self.W_sig_expected = ta.sig_to_tensor(expected_sig, dim = self.dim, order = self.order)
        self.W_sig_expected = W_sig.truncate(self.max_order_fn)

    @staticmethod
    def _sample(path, points=100):
        times = np.linspace(path[0, 0], path[-1, 0], points)
        idx = np.searchsorted(path[:, 0], times, side="left")
        #idx = np.unique(np.linspace(0, len(path) - 1, points).astype(int))


        return path[idx]

    def LF(self, F, max_order):
        dim = self.dim        
        LG = self.e1*self.env.b2 + self.e2*self.env.sigma2
        neg_LG = -LG

        max_iter = F.order #200
        
        exp_LG = LG.exp(MAX_ITER = max_iter, MAX_LEVEL = max_order, METHOD = '2')
        exp_neg_LG = neg_LG.exp(MAX_ITER = max_iter, MAX_LEVEL = max_order-1, METHOD = '2')
        
        LF = exp_LG*self.env.X0
        
        term1 = (exp_neg_LG.shuffle(self.e0*self.env.b0 + F*self.env.b1, max_order = max_order-1))*self.e1
        LF += exp_LG.shuffle(term1, max_order = max_order)
        term2 = (exp_neg_LG*self.env.sigma0)*self.e2
        LF += exp_LG.shuffle(term2, max_order = max_order)
        
        LF = LF.truncate(max_order)

        del LG
        del exp_LG
        del exp_neg_LG
        del term1
        del term2

        return LF

    def p_perf(self, F, max_order, return_running = False):
        max_order_x = max_order//2
        LF = self.LF(F, max_order_x)

        # Define functional of performance criterion
        p_perf = (LF.shuffle(LF, max_order = 2*max_order_x))*(self.e1*self.env.A)
        p_perf += (F.shuffle(F, max_order = 2*self.order))*(self.e1*self.env.B)
        p_perf += 2*self.env.C*LF*self.e1
        p_perf += 2*self.env.D*F*self.e1
        if return_running:
            p_perf_run = p_perf
        p_perf += (LF.shuffle(LF, max_order = 2*max_order_x))*(self.env.E)
        p_perf += 2*self.env.G*LF
        #p_perf = p_perf.truncate(max_order_fn+1)
        if return_running:
            return p_perf, p_perf_run
        else:
            return p_perf
        
    def value_fn(self, F, max_order_x = 3, max_order_fn = 3, truncate_fn = False, MAX_TAYLOR = 100):
        # if truncate_fn:
        #     max_order_x = max_order_fn//2
            
        p_perf = self.p_perf(F, max_order_fn)
        
        criterion = p_perf.dot(self.W_sig_expected)

        return criterion
        
    def train_sgd(self, n_epochs, lr=0.01, max_order_x = 3, max_order_fn = 3, method="SGD", truncate_fn = False, MAX_TAYLOR=100):
        
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
            loss = self.value_fn(self.F, max_order_x, max_order_fn, truncate_fn, MAX_TAYLOR)

            # Train
            loss.backward()
            optimiser.step()

            loss_val = loss.detach().numpy()
            pbar.set_description(str(loss_val))
            self.history.append(loss_val)