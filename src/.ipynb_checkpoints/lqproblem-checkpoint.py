from random import random
import numpy as np
from scipy.interpolate import CubicSpline
from tqdm import tqdm
from scipy.integrate import solve_ivp
import tensor_algebra as ta
import iisignature

def dM(t, T, M, U, V, R):
    M_matrix = M.reshape((2,2))
    diff  = (M_matrix.T@U@M_matrix + M_matrix@V + (M_matrix@V).T + R).flatten()
    return diff

def solve_riccati_implicit_timde_dep(
        T, nb_t, d,
        Q, Y, U, PT
):
    dt = T / (nb_t - 1)

    Pts = np.empty([nb_t, d, d])
    Pts[-1, ...] = PT

    # print(np.shape(Q))
    for t in range(nb_t - 2, -1, -1):
        if d == 1:
            Ptplusdt = np.array([np.squeeze(Pts[t + 1, ...])])
        else:
            Ptplusdt = np.squeeze(Pts[t + 1, ...])

        P = Ptplusdt.copy()
        for i in range(10):
            diff_big_mat = np.eye(d * d) + dt * (np.kron(Y[t].T, np.eye(d)) + np.kron(np.eye(d), Y[t].T) + np.kron(
                np.eye(d),
                np.transpose(U[t] @ P)
            ) + np.kron(P @ U[t], np.eye(d)))
            err = P + dt * (Q[t] + Y[t].T @ P + P @ Y[t] + P @ U[t] @ P) - Ptplusdt
            err = np.linalg.inv(diff_big_mat) @ err.flatten()
            P = P - err.reshape(d, d)
            
        P1 = Ptplusdt - dt * (Q[t+1] + Y[t+1].T @ Ptplusdt + Ptplusdt @ Y[t+1] + Ptplusdt @ U[t+1] @ Ptplusdt)
        P_avg = 0.5*(P1 + P)
        Pts[t, ...] = P_avg.copy()

    return Pts



class environment:
    """
    Model parameters for the environment.
    """
    def __init__(self, b0 = 1, b1 = 1e-3, b2 = 1, sigma0 = 1, sigma2 = 1, X0 = 10, A = 0, B = -1e-1, C = 0, D = 0, E = -1, G = 0, T = 1, Nt = 1_000):
        self.b0 = b0
        self.b0_tilde = self.b0 + 0.5*sigma2*sigma0
        self.b1 = b1
        self.b2 = b2
        self.b2_tilde = self.b2 + 0.5*sigma2**2
        self.sigma0 = sigma0
        self.sigma2 = sigma2
        self.X0 = X0
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.E = E
        self.G = G
        self.T = T
        self.Nt = Nt
        self.timesteps = np.linspace(0, self.T, num = (Nt+1))
        self.dim = 2

    def simulate_BM_with_drift(self, x0=0, mu=0, sigma=1, nsims=1):
        x = np.zeros((self.Nt+1, nsims))
        x[0,:] = x0
        dt = self.T/(self.Nt)
        errs = np.random.randn(self.Nt, nsims)
        for t in range(self.Nt):
            x[t + 1,:] = x[t,:] + dt * mu + np.sqrt(dt) * sigma * errs[t,:]
        return x   
        
    def simulate(self, nsims, strategy, order):
        dt = self.T/(self.Nt)
        self.X = np.zeros((self.Nt+1, nsims))
        self.W = self.simulate_BM_with_drift(x0=0, mu=0, sigma=1, nsims=nsims)
        self.W_sig = np.zeros(((self.dim**(order+1) - self.dim)//(self.dim - 1), nsims))
        self.nu = np.zeros((self.Nt+1, nsims))
        self.X[0,:] = self.X0
        self.run_crit = np.zeros((self.Nt+1, nsims))
        for it in range(self.Nt):            
            X_t = self.X[it,:]
            dW = self.W[it+1,:] - self.W[it, :]
            nu_t = strategy(self, self.dim, it, nsims)
            self.nu[it, :] = nu_t
            
            for nsim in range(nsims):
                path_increment = np.array([self.timesteps[it+1]-self.timesteps[it], dW[nsim]])
                self.W_sig[:, nsim] = iisignature.sigjoin(self.W_sig[:, nsim], path_increment, order)

            integrand = self.A*X_t**2 + self.B*nu_t**2 + 2*self.C*X_t + 2*self.D*nu_t
            self.run_crit[it+1,:] = self.run_crit[it,:] + integrand*dt
            
            self.X[it+1,:] = self.X[it,:] + (self.b0_tilde + self.b1*nu_t + self.b2_tilde*X_t)*dt + (self.sigma0 + self.sigma2*X_t)*dW
         
        self.nu[-1,:] = self.nu[-2,:]
        self.performance = self.run_crit[-1,:] + self.E*self.X[-1,:]**2 + 2*self.G*self.X[-1,:]
        
def simulate_BM_with_drift(env, x0=0, mu=0, sigma=1, nsims=1):
    x = np.zeros((env.Nt+1, nsims))
    x[0,:] = x0
    dt = env.T/(env.Nt)
    errs = np.random.randn(env.Nt, nsims)
    for t in range(env.Nt):
        x[t + 1,:] = x[t,:] + dt * mu + np.sqrt(dt) * sigma * errs[t,:]
    return x        
        
def expected_signature_bm(env, order = 3, nsims = 1_000):
    W = simulate_BM_with_drift(env, x0=0, mu=0, sigma=1, nsims=nsims)
    path_sig_sim = np.zeros((2**(order+1) - 2, nsims))
    for nsim in range(nsims):
        path = np.array([env.timesteps,W[:, nsim]]).T
        path_sig = iisignature.sig(path, order)
        path_sig_sim[:, nsim] = path_sig
    mean_sig = np.mean(path_sig_sim, axis=1)
    mean_sig = np.insert(mean_sig, 0, 1)
    mean_sig = ta.sig_to_tensor(mean_sig, dim = 2, order = order)
    return mean_sig
        
def sig_strag_dot(F, order = 3):
    def strategy(env, dim, it, nsims):
        strag = np.zeros((nsims,))
        for nsim in range(nsims):
            W_sig_flat = np.insert(env.W_sig[:, nsim], 0, 1)
            W_sig = ta.sig_to_tensor(W_sig_flat, dim, order)
            strag[nsim] = F.dot(W_sig)
        return strag
    return strategy

def sig_strag_dot_2(F, order = 3):
    def strategy(env, W_sig, dim, it):
        W_sig_flat = np.insert(W_sig[:], 0, 1)
        W_sig = ta.sig_to_tensor(W_sig_flat, dim, order)
        strag = F.dot(W_sig)
        return strag
    return strategy

def sig_strag_dot_1(F, order = 3):
    def strategy(env):
        nsims = env.W.shape[-1]
        strag = np.zeros((env.Nt+1, nsims))
        strag[0,:] = float(F[()])
        for nsim in range(nsims):
            path = np.array([env.timesteps, env.W[:,nsim]]).T
            W_sig_flat = iisignature.sig(path, order, 2)
            W_sig_flat = np.insert(W_sig_flat, 0, 1, axis=1)
            #innov_sig = ta.sig_to_tensor(innov_sig_flat, 2, order)
            strag[1:, nsim] = np.array([F.dot(ta.sig_to_tensor(W_sig_flat[it,:], 2, order)) for it in range(env.Nt)])
        return strag
    return strategy