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
    def __init__(self, b0 = 1, b1 = 1e-3, b2 = 1, b3 = 1e-5, sigma0 = 1, sigma2 = 1, k = 1e-3, theta1 = 1, theta2 = 1e-2, S0 = 100, Q0 = 10, X0 = 0, T = 1, Nt = 1_000):
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
        self.sigma0 = sigma0
        self.sigma2 = sigma2
        self.theta1 = theta1
        self.theta2 = theta2
        self.S0 = S0
        self.Q0 = Q0
        self.X0 = X0
        self.k = k
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
        
#         # computing the conditional variance of the filter
        
#         dP = lambda t, p: self.sigma_alpha**2 + 2*self.a1*p - (self.b1**2)*(p**2)/(self.sigma_s**2)
#         p_init = np.array([self.sigma_alpha_0])
#         sol = solve_ivp(fun = dP, 
#                         t_span = [0,self.T], 
#                         y0 = p_init,
#                         t_eval = self.timesteps,
#                         method = 'DOP853'
#                         )
#         self.V_alpha = sol.y.reshape(-1,)
#         self.V_alpha_eval = lambda t: CubicSpline(self.timesteps, self.V_alpha)(t)
        
#         # computing solution to Riccati eq associated with the HJB
        
#         self.U = np.zeros((2,2))
#         self.U[0,0] = self.a2**2/self.k
#         self.U[0,1] = self.a2/self.k
#         self.U[1,0] = self.U[0,1]
#         self.U[1,1] = 1/self.k
        
#         self.V = np.zeros((2,2))
#         self.V[0,0] = self.a1
#         self.V[1,0] = self.a2*(self.b2-2*self.theta1)/(2*self.k)
#         self.V[1,1] = (self.b2-2*self.theta1)/(2*self.k)
        
#         self.R = np.zeros((2,2))
#         self.R[0,1] = self.b1/2
#         self.R[1,0] = self.R[0,1]
#         self.R[1,1] = (self.b2 - 2*self.theta1)**2/(4*self.k) - self.theta2
        
#         self.dG2 = lambda t, G2: dM(t, self.T, G2, self.U, self.V, self.R)        
#         self.G2_terminal = np.zeros((2, 2))
#         self.G2 = solve_ivp(fun = self.dG2, 
#                         t_span = [0,self.T], 
#                         y0 = (self.G2_terminal).flatten(),
#                         t_eval = self.timesteps,
#                         method = 'DOP853').y[:,::-1]
#         self.G2_copy = np.array([((self.G2)[:,it]).reshape((2,2)) for it in range(self.Nt+1)])
#         self.h3 = self.G2_copy[:,0,0]
#         self.h4 = self.G2_copy[:,1,1]
#         self.h5 = self.G2_copy[:,0,1]
        
#         self.M1 = np.zeros((self.Nt+1,2,2))
#         self.M1[:,0,0] = (self.a1*self.k + self.a2**2*self.h3 + self.a2*self.h5)/self.k
#         self.M1[:,0,1] = (self.a2*self.h3 + self.h5)/self.k
#         self.M1[:,1,0] = self.a2*(self.b2 - 2*self.theta1 + 2*self.h4 + 2*self.a2*self.h5)/(2*self.k)
#         self.M1[:,1,1] = (self.b2 - 2*self.theta1 + 2*self.h4 + 2*self.a2*self.h5)/(2*self.k)
#         self.M1_eval = lambda t: np.array([[CubicSpline(self.timesteps, self.M1[:, i, j])(t).item() for j in range(2)] for i in range(2)])
        
#         self.M2 = np.zeros((self.Nt+1,2,1))
#         self.M2[:,0,0] = self.a0*self.h3
#         self.M2[:,1,0] = self.b0/2 + self.a0*self.h5
#         self.M2_eval = lambda t: np.array([[CubicSpline(self.timesteps, self.M2[:, i, 0])(t).item()] for i in range(2)])
        
#         dG1 = lambda t, G1: (self.M1_eval(self.T-t)@(G1.reshape((2,1))) + self.M2_eval(self.T-t)).flatten()
#         G1_init = np.zeros((2,1))
#         self.G1 = solve_ivp(fun = dG1, 
#                         t_span = [0,self.T], 
#                         y0 = G1_init.flatten(),
#                         t_eval = self.timesteps).y[:,::-1]
#         self.G1_eval = lambda t: np.array([[CubicSpline(self.timesteps, self.G1[i,:])(t)] for i in range(2)])
#         self.h1 = self.G1[0,:]
#         self.h2 = self.G1[1,:]               

    def simulate(self, nsims, strategy, order):
        dt = self.T/(self.Nt)
        self.S = np.zeros((self.Nt+1, nsims))
        self.Q = np.zeros((self.Nt+1, nsims))
        self.X = np.zeros((self.Nt+1, nsims))
        self.W = self.simulate_BM_with_drift(x0=0, mu=0, sigma=1, nsims=nsims)
        self.W_sig = np.zeros(((self.dim**(order+1) - self.dim)//(self.dim - 1), nsims))
        self.nu = np.zeros((self.Nt+1, nsims))
        self.S[0,:] = self.S0
        self.Q[0,:] = self.Q0
        self.X[0,:] = self.X0
        self.run_crit = np.zeros((self.Nt+1, nsims))
        for it in range(self.Nt):            
            S_t = self.S[it,:]
            Q_t = self.Q[it,:]
            dW = self.W[it+1,:] - self.W[it, :]
            nu_t = strategy(self, self.dim, it, nsims)
            self.nu[it, :] = nu_t
            
            for nsim in range(nsims):
                path_increment = np.array([self.timesteps[it+1]-self.timesteps[it], dW[nsim]])
                self.W_sig[:, nsim] = iisignature.sigjoin(self.W_sig[:, nsim], path_increment, order)
            
            dS_drift = -self.k*nu_t**2 + self.b0*Q_t + self.b1*nu_t*Q_t + self.b2*S_t*Q_t + self.b3*nu_t*S_t*Q_t + 0.5*Q_t*self.sigma2*S_t - self.theta2*Q_t**2 - 2*self.theta1*Q_t*nu_t
            dS_vol = self.sigma0*Q_t + self.sigma2*S_t*Q_t
            
            self.run_crit[it+1,:] = self.run_crit[it,:] + dS_drift*dt #+ dS_vol*dW
            
            self.S[it+1,:] = self.S[it,:] + (self.b0 + self.b1*nu_t + self.b2*S_t + self.b3*nu_t*S_t + 0.5*self.sigma2*S_t)*dt + (self.sigma0 + self.sigma2*S_t)*dW
            
            self.Q[it+1,:] = self.Q[it,:] + nu_t*dt
            self.X[it+1,:] = self.X[it,:] - (self.S[it,:] + self.k*nu_t)*nu_t*dt
         
        self.nu[-1,:] = self.nu[-2,:]
        self.wealth = self.X + self.Q*self.S
        
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
            W_sig_flat = np.insert(env.W_sig, 0, 1)
            W_sig = ta.sig_to_tensor(W_sig_flat, dim, order)
            strag[nsim] = F.dot(W_sig)
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