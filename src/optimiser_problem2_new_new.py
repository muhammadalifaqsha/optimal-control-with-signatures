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
        F_dim = len(F)
        dim = self.dim
        F = ta.sig_to_tensor(F, self.env.dim, self.order)
        
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
        criterion = p_perf.dot(self.W_sig_expected)
        
        self.p_perf = p_perf

        return -float(criterion)


    def train(self, trunc_lev_s = 3, method="BFGS", **kwargs):
        F = self.F.flatten()    
        #F = np.zeros(len(self.F.flatten()))

        pbar = tqdm(desc="Optimising")
        t = time()
        res = optimize.minimize(self.value_fn, F, args=(trunc_lev_s,), method=method,
                                callback=lambda _ : pbar.update(), **kwargs)
        dt = time() - t
        pbar.close()

        F_dim = len(res.x)
        F = res.x
        self.F = ta.sig_to_tensor(F, self.dim, self.order)

        print("Done in {} seconds. Maximum value: {}".format(dt, -res.fun))
        
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
            trunc_lev = trunc_lev_s
            F = self.F

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
            loss = -p_perf.dot(self.W_sig_expected)

            # Train
            loss.backward()
            optimiser.step()

            loss_val = -loss.detach().numpy()
            pbar.set_description(str(loss_val))
            self.history.append(loss_val)




    def plot_results(self, paths, show=True, name1="Equity", name2="FX", return_all=False, return_qty=False):
        paths = np.array([self._sample(path, points=self.params["n_orders"]) for path in paths])
        #paths[:, :, 1] *= self.params["s0"]


        all_stock = []
        all_fx = []
        wealth = []
        all_pos_fx = []
        all_pos_eq = []
        all_wealth = []
        dt = 1. / self.params["n_orders"]
        for path in tqdm(paths):
            #times = np.linspace(path[0, 0], path[-1, 0], self.params["n_orders"])
            #dt = times[1]
            #idx = np.searchsorted(path[:, 0], times, side="left")
            #path = path[idx]

            stock = [self.params["q0"] / self.params["s0"]]
            fx = [0.]
            _pos_fx = []
            _pos_eq = []

            w = 0.
            wealth_path = [w]
            for i in range(1, len(path)):
                #if path[i, 0] < times[0]:
                #    continue
                #times = times[1:]
                #dt = path[i, 0] - path[i - 1, 0]

                if self.params["N"] > 0:
                    sig = np.r_[1., iisignature.sig(path[:i + 1], self.params["N"])]
                else:
                    sig = np.array([1.])

                pos_stock = np.dot(self.l1.flatten(), sig)
                pos_fx = np.dot(self.l2.flatten(), sig)

                _pos_eq.append(pos_stock)
                _pos_fx.append(pos_fx)

                q = stock[-1] - pos_stock * dt
                q_fx = fx[-1] - pos_fx * dt + pos_stock * (path[i, 1] - self.params["Lambda1"] * pos_stock) * dt
                w += pos_fx * (path[i, 2] - self.params["Lambda2"] * pos_fx) * dt
                
                wealth_path.append(w)
                stock.append(q)
                fx.append(q_fx)

            all_stock.append(self.params["s0"] * np.array(stock))
            all_fx.append(self.params["s0"] * np.array(fx))
            all_pos_fx.append(_pos_fx)
            all_pos_eq.append(_pos_eq)
            all_wealth.append(wealth_path)

            w += fx[-1] * (path[i, 2] - self.params["alpha2"] * fx[-1])
            wealth.append(self.params["s0"] ** 2 * w)

        timeline = np.linspace(0, 1, len(all_stock[0]))
        for stock in all_stock:
            plt.plot(timeline, stock, "b", alpha=0.1)

        for fx in all_fx:
            plt.plot(timeline, fx, "r", alpha=0.1)




        legend_elements = [Line2D([0], [0], color='b', lw=4, label='{} inventory'.format(name1)),
                           Line2D([0], [0], color='r', lw=4, label='{} inventory'.format(name2))]

        plt.legend(handles=legend_elements)


        plt.title("SIGNATURES\nAverage wealth: {}".format(np.mean(wealth)))
        plt.xlabel("Time")
        plt.ylabel("Inventory")

        if show:
            plt.show()

        if return_all:
            return wealth, np.array(all_pos_eq), np.array(all_pos_fx)

        if return_qty:
            return wealth, np.array(all_stock), np.array(all_fx), np.array(all_wealth)

        return wealth

    @staticmethod
    def benchmark(t, equity, fx, phi1=0., phi2=0., Lambda1=0., Lambda2=0.,
                  alpha1=0., alpha2=0., q0=1., q0_fx=1., s0=1., **kwargs):
        """Constant executing of stock during first half, then constant executing in FX."""

        q0 /= s0

        phi1 += 1e-8
        phi2 += 1e-8
        gamma1 = np.sqrt(phi1 / Lambda1)
        gamma2 = np.sqrt(phi2 / Lambda2)
        dseta1 = (alpha1 + np.sqrt(Lambda1 * phi1)) / (alpha1 - np.sqrt(Lambda1 * phi1))
        dseta2 = (alpha2 + np.sqrt(Lambda2 * phi2)) / (alpha2 - np.sqrt(Lambda2 * phi2))

        threshold = 0.5


        if t <= threshold:
            s = q0 / (threshold + Lambda1 / alpha1)
            return s, 0

        s = q0_fx / (1 - threshold + Lambda2 / alpha2)

        return 0, s


    def plot_benchmark(self, paths, show=True, **kwargs):
        paths = np.array([self._sample(path, points=self.params["n_orders"]) for path in paths])

        all_stock = []
        all_fx = []
        wealth = []
        dt = 1. / self.params["n_orders"]

        for path in tqdm(paths):
            #times = np.linspace(path[0, 0], path[-1, 0], self.params["n_orders"])
            #dt = times[1]
            #idx = np.searchsorted(path[:, 0], times, side="left")
            #path = path[idx]

            stock = [self.params["q0"] / self.params["s0"]]
            fx = [0.]
            w = 0.
            q0_fx = None
            for i in range(1, len(path)):
                #if path[i, 0] < times[0]:
                #    continue
                #times = times[1:]
                #dt = path[i, 0] - path[i - 1, 0]

                if path[i, 0] > 0.5 and q0_fx is None:
                    q0_fx = fx[-1]

                pos_stock, pos_fx = self.benchmark(*path[i, :3], q0_fx=q0_fx, **kwargs)

                q = stock[-1] - pos_stock * dt
                q_fx = fx[-1] - pos_fx * dt + pos_stock * (path[i, 1] - self.params["Lambda1"] * pos_stock) * dt
                w += pos_fx * (path[i, 2] - self.params["Lambda2"] * pos_fx) * dt

                stock.append(q)
                fx.append(q_fx)

            all_stock.append(self.params["s0"] * np.array(stock))
            all_fx.append(self.params["s0"] * np.array(fx))


            w += fx[-1] * (path[i, 2] - self.params["alpha2"] * fx[-1])
            wealth.append(self.params["s0"] ** 2 * w)



        timeline = np.linspace(0, 1, len(all_stock[0]))
        for stock in all_stock:
            plt.plot(timeline, stock, "b", alpha=0.1)

        for fx in all_fx:
            plt.plot(timeline, fx, "r", alpha=0.1)

        legend_elements = [Line2D([0], [0], color='b', lw=4, label='Equity inventory'),
                           Line2D([0], [0], color='r', lw=4, label='FX inventory')]

        plt.legend(handles=legend_elements)

        plt.title("BENCHMARK\nAverage wealth: {}".format(np.mean(wealth)))
        plt.xlabel("Time")
        plt.ylabel("Inventory")

        if show:
            plt.show()

        return wealth

    def plot_twap(self, paths, show=True, direct=False, **kwargs):
        paths = np.array([self._sample(path, points=self.params["n_orders"]) for path in paths])

        dt = 1. / self.params["n_orders"]



        all_stock = []
        wealth = []
        for path in tqdm(paths):
            #times = np.linspace(path[0, 0], path[-1, 0], self.params["n_orders"])
            #dt = times[1]
            #idx = np.searchsorted(path[:, 0], times, side="left")
            #path = path[idx]


            Q = [self.params["q0"] / self.params["s0"]]
            w = 0.

            for i in range(1, len(path)):
                #if path[i, 0] < times[0]:
                #    continue
                #times = times[1:]

                pos = Q[0]

                q = Q[-1] - pos * dt
                q_fx = pos * (path[i, 1] - self.params["Lambda1"] * pos) * dt

                if direct:
                    w += q_fx
                else:
                    w += q_fx * (path[i, 2] - self.params["alpha2"] * q_fx)# * dt



                Q.append(q)

            all_stock.append(self.params["s0"] * np.array(Q))



            wealth.append(self.params["s0"] ** 2 * w)




        timeline = np.linspace(0, 1, len(all_stock[0]))
        for stock in all_stock:
            plt.plot(timeline, stock, "b", alpha=0.1)

        legend_elements = [Line2D([0], [0], color='b', lw=4, label='Inventory')]

        plt.legend(handles=legend_elements)

        plt.title("TWAP\nAverage wealth: {}".format(np.mean(wealth)))
        plt.xlabel("Time")
        plt.ylabel("Inventory")

        if show:
            plt.show()

        return wealth



    def export(self, name, paths):
        now = datetime.datetime.today()
        path = "../logs/{} {}".format(name, now)

        if not os.path.exists(path):
            os.makedirs(path)

        with open(os.path.join(path, "metadata"), "w") as f:
            f.write(str(self.params))

        wealth_signatures = self.plot_results(paths, show=False)
        wealth_benchmark = self.plot_benchmark(paths, show=False, **self.params)
        plt.close()

        # Export savings per share plot

        plt.figure(figsize=(14, 4))

        plt.subplot(1, 2, 1)
        _ = plt.hist([wealth_signatures, wealth_benchmark], label=("Signatures", "Benchmark"))
        plt.legend()
        plt.xlabel("Wealth")

        plt.subplot(1, 2, 2)
        savings_per_share = 1e4 * np.subtract(wealth_signatures, wealth_benchmark) / wealth_benchmark
        _ = plt.hist(savings_per_share, bins=15)
        plt.title("Average: {}".format(np.mean(savings_per_share)))
        plt.xlabel("Savings per share (in bp)")
        plt.savefig(os.path.join(path, "image.png"))
        plt.close()

        # Export sagins per share
        np.savetxt(os.path.join(path, "wealth_signatures"), wealth_signatures)
        np.savetxt(os.path.join(path, "wealth_benchmark"), wealth_benchmark)

        with open(os.path.join(path, "results"), "w") as f:
            f.write("Average: {}\n".format(np.mean(savings_per_share)))
            f.write("Median: {}\n".format(np.median(savings_per_share)))
            f.write("std: {}\n".format(np.std(savings_per_share)))
            f.write("Min: {}\n".format(np.min(savings_per_share)))
            f.write("Max: {}\n".format(np.max(savings_per_share)))


