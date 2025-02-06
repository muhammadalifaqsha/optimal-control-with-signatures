import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import datetime
from time import time
from scipy import optimize

import torch
from torch import optim
torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)

import iisignature

import tensor_algebra as ta

class Optimiser:
    """Solves the optimisation problem on the tensor algebra."""

    def __init__(self, ES, N, Lambda1, Lambda2, q0, alpha1, alpha2,
                 phi1, phi2, s0=1, dim=2, n_orders=100, **kwargs):
        self.params = {
                        "Lambda1":    Lambda1,
                        "Lambda2":    Lambda2,
                        "N":          N,
                        "q0":         q0,
                        "alpha1":     alpha1,
                        "alpha2":     alpha2,
                        "phi1":       phi1,
                        "phi2":       phi2,
                        "s0":         s0,
                        "n_orders":   n_orders
                      }

        self.dim = dim
        self.ES = ES
        self.history = []

        # Define the linear functionals we'll be solving for.
        self.l1 = ta.Tensor(self.dim, self.params["N"])
        self.l1.value = [torch.randn([self.dim] * n, requires_grad=True) for n in range(self.params["N"] + 1)]

        self.l2 = ta.Tensor(self.dim, self.params["N"])
        self.l2.value = [torch.randn([self.dim] * n, requires_grad=True) for n in range(self.params["N"] + 1)]



    @staticmethod
    def _sample(path, points=100):
        times = np.linspace(path[0, 0], path[-1, 0], points)
        idx = np.searchsorted(path[:, 0], times, side="left")
        #idx = np.unique(np.linspace(0, len(path) - 1, points).astype(int))


        return path[idx]

    def value_fn(self, l, s01, s02):
        l_dim = len(l)
        l1, l2 = l[:l_dim // 2], l[l_dim // 2 :]
        l1 = ta.sig_to_tensor(l1, self.dim, self.params["N"])
        l2 = ta.sig_to_tensor(l2, self.dim, self.params["N"])

        # Define tensor algebra constants.
        ONE = ta.one(self.dim)
        E1 = ta.e(0, self.dim)
        E2 = ta.e(1, self.dim)
        E3 = ta.e(2, self.dim)

        # Define spot prices.


        # Define polynomial on the dual of the tensor algebra

        gl1 = self.params["Lambda1"] * l1
        gl2 = self.params["Lambda2"] * l2

        pl1 = E2 + s01 * ONE
        pl2 = E3 + s02 * ONE
        ptildel1 = E2 + s01 * ONE - gl1
        ptildel2 = E3 + s02 * ONE - gl2


        wl1 = ptildel1.shuffle(l1) * E1
        wl2 = ptildel2.shuffle(l2) * E1

        ql1 = (self.params["q0"] / self.params["s0"]) * ONE - l1 * E1
        ql2 = -l2 * E1

        ml1 = ql1.shuffle(pl1)

        poly = wl2
        poly += ml1.shuffle(pl2 - self.params["alpha1"] * ml1)
        poly += (ql2 + wl1).shuffle(pl2 - self.params["alpha2"] * ql2 - self.params["alpha2"] * wl1)

        if self.params["phi1"] != 0.:
            # L2 penalty on stock inventory
            poly += -self.params["phi1"] * ql1.shuffle(ql1) * E1

        if self.params["phi2"] != 0.:
            # L2 penalty on foreign currency inventory
            poly += -self.params["phi2"] * (ql2 + wl1).shuffle(ql2 + wl1) * E1

        # Define loss function
        loss = -poly.dot(self.ES)

        return float(loss)


    def train(self, paths, method="BFGS", **kwargs):
        l = np.zeros(2 * len(self.l1.flatten()))
        s01 = paths[0][0, 1]
        s02 = paths[0][0, 2]


        pbar = tqdm(desc="Optimising")
        t = time()
        res = optimize.minimize(self.value_fn, l, args=(s01, s02,), method=method,
                                callback=lambda _ : pbar.update(), **kwargs)
        dt = time() - t
        pbar.close()

        l_dim = len(res.x)
        l1, l2 = res.x[:l_dim // 2], res.x[l_dim // 2 :]
        self.l1 = ta.sig_to_tensor(l1, self.dim, self.params["N"])
        self.l2 = ta.sig_to_tensor(l2, self.dim, self.params["N"])

        print("Done in {} seconds. Maximum value: {}".format(dt, -res.fun))

    def train_sgd(self, paths, n_epochs, lr=0.01, method="SGD"):
        # Define tensor algebra constants.
        ONE = ta.one(self.dim)
        E1 = ta.e(0, self.dim)
        E2 = ta.e(1, self.dim)
        E3 = ta.e(2, self.dim)

        # Define spot prices.
        s01 = paths[0][0, 1]
        s02 = paths[0][0, 2]

        # Define Torch optimiser.
        if method == "SGD":
            optimiser = optim.SGD(self.l1.value + self.l2.value, lr=lr)
        elif method == "Adam":
            optimiser = optim.Adam(self.l1.value + self.l2.value, lr=lr)
        else:
            raise ValueError("Method {} not supported.".format(method))

        pbar = tqdm(range(n_epochs))

        for _ in pbar:
            optimiser.zero_grad()

            # Define polynomial on the dual of the tensor algebra

            gl1 = self.params["Lambda1"] * self.l1
            gl2 = self.params["Lambda2"] * self.l2

            pl1 = E2 + s01 * ONE
            pl2 = E3 + s02 * ONE
            ptildel1 = E2 + s01 * ONE - gl1
            ptildel2 = E3 + s02 * ONE - gl2


            wl1 = ptildel1.shuffle(self.l1) * E1
            wl2 = ptildel2.shuffle(self.l2) * E1

            ql1 = (self.params["q0"] / self.params["s0"]) * ONE - self.l1 * E1
            ql2 = -self.l2 * E1

            ml1 = ql1.shuffle(pl1)

            poly = wl2
            poly += ml1.shuffle(pl2 - self.params["alpha1"] * ml1)
            poly += (ql2 + wl1).shuffle(pl2 - self.params["alpha2"] * ql2 - self.params["alpha2"] * wl1)

            if self.params["phi1"] != 0.:
                # L2 penalty on stock inventory
                poly += -self.params["phi1"] * ql1.shuffle(ql1) * E1

            if self.params["phi2"] != 0.:
                # L2 penalty on foreign currency inventory
                poly += -self.params["phi2"] * (ql2 + wl1).shuffle(ql2 + wl1) * E1


            # Define loss function
            loss = -poly.dot(self.ES)

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


