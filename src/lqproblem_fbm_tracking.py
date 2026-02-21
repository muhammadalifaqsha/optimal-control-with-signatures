from random import random
import numpy as np
from scipy.interpolate import CubicSpline
from tqdm.auto import tqdm
from math import factorial
from scipy.integrate import solve_ivp
import tensor_algebra as ta
import iisignature
from fbm import FBM
from differint.differint import RL

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
    def __init__(self, b0 = 0, b1 = 1e-3, b2 = 1, sigma0 = 1, sigma2 = 0, X0 = 10, A = 0, B = 1e-1, C = 0, D = 0, E = 1, G = 0, T = 1, Nt = 1_000, hurst_inv = 4):
        # b0, sigma2, C, D, G have to be zero in this case
        self.b0 = 0
        self.b0_tilde = self.b0 + 0.5*sigma2*sigma0
        self.b1 = b1
        self.b2 = b2
        self.b2_tilde = self.b2 + 0.5*sigma2**2
        self.sigma0 = sigma0
        self.sigma2 = 0
        self.X0 = X0
        self.A = A
        self.B = B
        self.C = 0
        self.D = 0
        self.E = E
        self.G = 0
        self.T = T
        self.Nt = Nt
        self.hurst_inv = hurst_inv
        self.timesteps = np.linspace(0, self.T, num = (Nt+1))
        self.dim = 2
        self.first_time_run_sim = True
        self.nsims = None
        self.W = None
        self.optimal_calculated = False
        self.f2 = None
        self.phi = None
        self.K_fbm_integrand = None

    def simulate_BM_with_drift(self, x0=0, mu=0, sigma=1, nsims=1):
        x = np.zeros((self.Nt+1, nsims))
        x[0,:] = x0
        dt = self.T/(self.Nt)
        errs = np.random.randn(self.Nt, nsims)
        for t in range(self.Nt):
            x[t + 1,:] = x[t,:] + dt * mu + np.sqrt(dt) * sigma * errs[t,:]
        return x   

    def BM_signature(self, W, order):
        nsims = W.shape[1]
        W_sig = np.zeros((self.Nt+1, (self.dim**(order+1) - self.dim)//(self.dim - 1), nsims))
        for it in range(self.Nt):
            dW = W[it+1,:] - W[it, :]
            for nsim in range(nsims):
                path_increment = np.array([self.timesteps[it+1]-self.timesteps[it], dW[nsim]])
                W_sig[it+1, :, nsim] = iisignature.sigjoin(W_sig[it, :, nsim], path_increment, order)
        return W_sig
        
    def simulate(self, nsims, strategy, order, fast = False):
        # (fast = True) is for computing the signatures of W (e.g. for signature-based control)
        dt = self.T/(self.Nt)
        self.X = np.zeros((self.Nt+1, nsims))
        if nsims != self.nsims:
            self.first_time_run_sim = True
        if self.first_time_run_sim:
            self.nsims = nsims
            print("Simulating fractional Brownian noise")
            self.W = np.array([(FBM(n=self.Nt, hurst=1/self.hurst_inv, length=self.T, method='daviesharte')).fbm() for i in range(nsims)]).T
            if not fast:
                print("Calculating the historical signatures of fractional Brownian noise")
                self.W_sig_hist = self.BM_signature(self.W, order)
            self.first_time_run_sim = False
        if not fast:
            self.W_sig = np.zeros(((self.dim**(order+1) - self.dim)//(self.dim - 1), nsims))
        self.nu = np.zeros((self.Nt+1, nsims))
        self.X[0,:] = self.X0
        self.run_crit = np.zeros((self.Nt+1, nsims))
        self.nu[0, :] = strategy(self, self.dim, 0, nsims)

        print("Simulation progress")
        with tqdm(total=self.Nt) as pbar:
            for it in range(self.Nt):            
                X_t = self.X[it,:]
                self.X_t = X_t
                dW = self.W[it+1,:] - self.W[it, :]
                nu_t = strategy(self, self.dim, it, nsims)
                self.nu[it, :] = nu_t
    
                if not fast:
                    self.W_sig = self.W_sig_hist[it, :, :]
    
                integrand = self.A*X_t**2 + self.B*nu_t**2 + 2*self.C*X_t + 2*self.D*nu_t
                self.run_crit[it+1,:] = self.run_crit[it,:] + integrand*dt
                
                self.X[it+1,:] = self.X[it,:] + (self.b0_tilde + self.b1*nu_t + self.b2_tilde*X_t)*dt + (self.sigma0 + self.sigma2*X_t)*dW
                pbar.update(1)
         
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

def expected_signature_fbm(env, order = 3, nsims = 1_000):
    W = np.array([(FBM(n=env.Nt, hurst=1/env.hurst_inv, length=env.T, method='daviesharte')).fbm() for i in range(nsims)]).T
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

def sig_strag_dot_vec(F, order = 3):
    def strategy(env, dim, it, nsims):
        strag = np.zeros((nsims,))
        F_flat = F.flatten()
        F_flat = F_flat.reshape((1,-1))
        dim_tensor = (env.dim**(order+1) - env.dim)//(env.dim - 1)
        if dim_tensor == 0:
            strag = F_flat[:, 0]
        else:
            W_sig_trunc = env.W_sig[:dim_tensor, :]
            strag = (F_flat[:, 1:])@W_sig_trunc
            strag = strag + F_flat[:, 0]
            strag = strag.reshape((-1,))
        return strag
    return strategy

def strategy_optimal_feedback(env, dim, it, nsims):
    if not env.optimal_calculated:        
        f2_terminal = np.array([env.E])
        df2 = lambda t, g: -env.b1**2*g**2/(env.B) + (env.sigma2**2+2*env.b2_tilde)*g + env.A
        f2 = solve_ivp(fun = df2, 
                        t_span = [0,env.T], 
                        y0 = f2_terminal,
                        t_eval = env.timesteps,
                        method = 'DOP853').y[:,::-1]    
        f1_terminal = np.array([2*env.G])
        df1 = lambda t, g: env.C - env.b1*env.D*f2[0,int(((env.T-t)*env.Nt)//env.T)]/env.B + env.sigma0*env.sigma2*f2[0,int(((env.T-t)*env.Nt)//env.T)] - env.b1**2*g*f2[0,int(((env.T-t)*env.Nt)//env.T)]/env.B + f2[0,int(((env.T-t)*env.Nt)//env.T)]*env.b0_tilde + g*env.b2_tilde
        f1 = solve_ivp(fun = df1, 
                        t_span = [0,env.T], 
                        y0 = f1_terminal,
                        t_eval = env.timesteps,
                        method = 'DOP853').y[:,::-1]

        phi = np.zeros((env.Nt+1, env.Nt+1))
        phi[0,0] = 1
        
        for i in range(env.Nt):
            phi[i+1,i+1] = 1
            if i < env.Nt-1:
                dphi_i_forward = lambda t, p: -(env.b2_tilde - f2[0, int((t*env.Nt)//env.T)]*env.b1**2/(env.B))*p
                phi_i_forward = solve_ivp(fun = dphi_i_forward,
                                          t_span = [env.timesteps[i+1], env.T],
                                          y0 = np.array([1]),
                                          t_eval = env.timesteps[i+1:],
                                          method = 'DOP853').y
                phi[i+1, i+1:] = phi_i_forward[0, :]

            dphi_i_backward = lambda t, p: (env.b2_tilde - f2[0, int(((env.T-t)*env.Nt)//env.T)]*env.b1**2/(env.B))*p
            phi_i_backward = solve_ivp(fun = dphi_i_backward,
                                       t_span = [0, env.timesteps[i+1]],
                                       y0 = np.array([1]),
                                       t_eval = env.timesteps[:i+2],
                                       method = 'DOP853').y[:,::-1]
            phi[i+1, :i+2] = phi_i_backward[0, :]

        env.phi = phi  
        env.f2 = f2
        
        K_fbm_integrand = np.zeros((env.Nt+1, env.Nt))

        print("Calculating integrand to fractional Brownian motion")
        with tqdm(total=env.Nt) as pbar:
            for j in range(env.Nt):
                func_1 = env.timesteps**(1/env.hurst_inv - 1/2)*phi[:, j+1]*f2[0,:]
                func_1[:j+1] = np.zeros((j+1,))
                I_func_1 = RL(1/2 - 1/env.hurst_inv, func_1[::-1], domain_start=0, domain_end=env.T, num_points=env.Nt+1)
                I_func_1 = I_func_1[::-1]
                I_func_1_cropped = I_func_1[:j+2]
                I_func_2 = RL(1/env.hurst_inv - 1/2, I_func_1_cropped[::-1], domain_start=0, domain_end=env.timesteps[j+1], num_points=j+2)
                I_func_2 = I_func_2[::-1]
                K_fbm_integrand[j+1, :j+1] = (env.timesteps[:j+1])**(1/2-1/env.hurst_inv)*I_func_2[:j+1]
                pbar.update(1)

        env.K_fbm_integrand = K_fbm_integrand

        env.optimal_calculated = True
        
    strag = (-env.b1*env.f2[0,it]*env.X[it,:])/(env.B)
    
    if it > 0:
        W_diff = env.W[1:it+1, :] - env.W[:it, :]
        Kt = env.K_fbm_integrand[it, :it]@W_diff
        strag = strag - env.b1*Kt/(env.B)
        
    return strag