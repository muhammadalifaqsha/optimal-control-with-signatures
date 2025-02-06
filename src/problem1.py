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
    def __init__(self, a0 = 0, a1 = -1, a2 = 0, b0 = 0, b1 = 1, b2 = 1, sigma_alpha = 1, sigma_alpha_0 = 0, sigma_s = 1, k = 1e-3, theta1 = 1, theta2 = 1e-2, alpha0 = 0, S0 = 100, Q0 = 0, X0 = 0, T = 1, Nt =1_000):
        self.a0 = a0
        self.a1 = a1
        self.a2 = a2
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2
        self.theta1 = theta1
        self.theta2 = theta2
        self.alpha0 = alpha0
        self.S0 = S0
        self.Q0 = Q0
        self.X0 = X0
        self.sigma_alpha = sigma_alpha
        self.sigma_alpha_0 = sigma_alpha_0
        self.sigma_s = sigma_s
        self.k = k
        self.T = T
        self.Nt = Nt
        self.timesteps = np.linspace(0, self.T, num = (Nt+1))
        self.dim = 2
        
        # computing the conditional variance of the filter
        
        dP = lambda t, p: self.sigma_alpha**2 + 2*self.a1*p - (self.b1**2)*(p**2)/(self.sigma_s**2)
        p_init = np.array([self.sigma_alpha_0])
        sol = solve_ivp(fun = dP, 
                        t_span = [0,self.T], 
                        y0 = p_init,
                        t_eval = self.timesteps,
                        method = 'DOP853'
                        )
        self.V_alpha = sol.y.reshape(-1,)
        self.V_alpha_eval = lambda t: CubicSpline(self.timesteps, self.V_alpha)(t)
        
        # computing solution to Riccati eq associated with the HJB
        
        self.U = np.zeros((2,2))
        self.U[0,0] = self.a2**2/self.k
        self.U[0,1] = self.a2/self.k
        self.U[1,0] = self.U[0,1]
        self.U[1,1] = 1/self.k
        
        self.V = np.zeros((2,2))
        self.V[0,0] = self.a1
        self.V[1,0] = self.a2*(self.b2-2*self.theta1)/(2*self.k)
        self.V[1,1] = (self.b2-2*self.theta1)/(2*self.k)
        
        self.R = np.zeros((2,2))
        self.R[0,1] = self.b1/2
        self.R[1,0] = self.R[0,1]
        self.R[1,1] = (self.b2 - 2*self.theta1)**2/(4*self.k) - self.theta2
        
        self.dG2 = lambda t, G2: dM(t, self.T, G2, self.U, self.V, self.R)        
        self.G2_terminal = np.zeros((2, 2))
        self.G2 = solve_ivp(fun = self.dG2, 
                        t_span = [0,self.T], 
                        y0 = (self.G2_terminal).flatten(),
                        t_eval = self.timesteps,
                        method = 'DOP853').y[:,::-1]
        self.G2_copy = np.array([((self.G2)[:,it]).reshape((2,2)) for it in range(self.Nt+1)])
        self.h3 = self.G2_copy[:,0,0]
        self.h4 = self.G2_copy[:,1,1]
        self.h5 = self.G2_copy[:,0,1]
        
        self.M1 = np.zeros((self.Nt+1,2,2))
        self.M1[:,0,0] = (self.a1*self.k + self.a2**2*self.h3 + self.a2*self.h5)/self.k
        self.M1[:,0,1] = (self.a2*self.h3 + self.h5)/self.k
        self.M1[:,1,0] = self.a2*(self.b2 - 2*self.theta1 + 2*self.h4 + 2*self.a2*self.h5)/(2*self.k)
        self.M1[:,1,1] = (self.b2 - 2*self.theta1 + 2*self.h4 + 2*self.a2*self.h5)/(2*self.k)
        self.M1_eval = lambda t: np.array([[CubicSpline(self.timesteps, self.M1[:, i, j])(t).item() for j in range(2)] for i in range(2)])
        
        self.M2 = np.zeros((self.Nt+1,2,1))
        self.M2[:,0,0] = self.a0*self.h3
        self.M2[:,1,0] = self.b0/2 + self.a0*self.h5
        self.M2_eval = lambda t: np.array([[CubicSpline(self.timesteps, self.M2[:, i, 0])(t).item()] for i in range(2)])
        
        dG1 = lambda t, G1: (self.M1_eval(self.T-t)@(G1.reshape((2,1))) + self.M2_eval(self.T-t)).flatten()
        G1_init = np.zeros((2,1))
        self.G1 = solve_ivp(fun = dG1, 
                        t_span = [0,self.T], 
                        y0 = G1_init.flatten(),
                        t_eval = self.timesteps).y[:,::-1]
        self.G1_eval = lambda t: np.array([[CubicSpline(self.timesteps, self.G1[i,:])(t)] for i in range(2)])
        self.h1 = self.G1[0,:]
        self.h2 = self.G1[1,:]               

    def simulate(self, nsims, strategy):
        dt = self.T/(self.Nt)
        alpha = np.zeros((self.Nt+1, nsims))
        alpha_hat = np.zeros((self.Nt+1, nsims))
        S = np.zeros((self.Nt+1, nsims))
        Q = np.zeros((self.Nt+1, nsims))
        X = np.zeros((self.Nt+1, nsims))
        nu = np.zeros((self.Nt+1, nsims))
        innov = np.zeros((self.Nt+1, nsims))
        alpha[0,:] = self.alpha0
        alpha_hat[0,:] = self.alpha0
        S[0,:] = self.S0
        Q[0,:] = self.Q0
        X[0,:] = self.X0
        run_risk = np.zeros((self.Nt+1, nsims))
        for it in range(self.Nt):
            self.S = S
            self.alpha = alpha
            self.alpha_hat = alpha_hat
            self.innov = innov
            self.Q = Q
            self.X = X
            
            S_t = S[it,:]
            alpha_t = alpha[it,:]
            alpha_hat_t = alpha_hat[it,:]
            nu_t = strategy(self, it)
            nu[it, :] = nu_t
            
            self.alpha_t = alpha_t
            self.nu_t = nu_t
            
            run_risk[it+1,:] = run_risk[it,:] + self.theta2*(Q[it,:]**2)*dt
            
            S[it+1,:] = S[it,:] + (self.b0 + self.b1*alpha_t + self.b2*nu_t)*dt + self.sigma_s*np.random.randn(1, nsims)*np.sqrt(dt)
            
            innov[it+1,:] = innov[it,:]
            innov[it+1,:] += ((S[it+1,:] - S[it,:])/self.sigma_s)
            innov[it+1,:] -= ((self.b0 + self.b1*alpha_hat_t + self.b2*nu_t)*dt/self.sigma_s)
            
            alpha[it+1,:] = alpha[it,:]
            alpha[it+1,:] += (self.a0 + self.a1*alpha_t + self.a2*nu_t)*dt
            alpha[it+1,:] += self.sigma_alpha*np.random.randn(nsims)*np.sqrt(dt)
            
            alpha_hat[it+1,:] = alpha_hat[it,:]
            alpha_hat[it+1,:] += (self.a0 + self.a1*alpha_hat_t + self.a2*nu_t)*dt
            alpha_hat[it+1,:] += (self.b1*self.V_alpha[it]/self.sigma_s)*(innov[it+1,:] - innov[it,:])
            
            Q[it+1,:] = Q[it,:] + nu_t*dt
            X[it+1,:] = X[it,:] - (S[it,:] + self.k*nu_t)*nu_t*dt
         
        nu[-1,:] = nu[-2,:]
        self.nu = nu
        self.wealth = self.X + self.Q*self.S
        self.crit = self.wealth - self.theta1*self.Q**2 - run_risk
        
    def optimal_strag(self, env, it):
        alpha_hat_t = env.alpha_hat[it,:]
        Q_t = env.Q[it,:]
        nu_opt = alpha_hat_t*(self.a2*self.h3[it] + self.h5[it])/self.k
        nu_opt += Q_t*(self.b2 - 2*self.theta1 + 2*self.h4[it] + 2*self.a2*self.h5[it])/(2*self.k)
        nu_opt += (self.a2*self.h1[it] + self.h2[it])/self.k
        
        return nu_opt
        
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
    def strategy(env, it):
        if it == 0:
            strag = float(F[()])
        else:
            innov_stopped = env.innov[:(it+1),:]
            nsims = innov_stopped.shape[-1]
            timesteps_stopped = env.timesteps[:(it+1)]
            strag = np.zeros((nsims,))
            for nsim in range(nsims):
                path = np.array([timesteps_stopped,innov_stopped[:, nsim]]).T
                innov_sig_flat = iisignature.sig(path, order)
                innov_sig_flat = np.insert(innov_sig_flat, 0, 1)
                innov_sig = ta.sig_to_tensor(innov_sig_flat, 2, order)
                strag[nsim] = F.dot(innov_sig)
        return strag
    return strategy

def sig_strag_dot_1(F, order = 3):
    def strategy(env):
        nsims = env.innov.shape[-1]
        strag = np.zeros((env.Nt+1, nsims))
        strag[0,:] = float(F[()])
        for nsim in range(nsims):
            path = np.array([env.timesteps, env.innov[:,nsim]]).T
            innov_sig_flat = iisignature.sig(path, order, 2)
            innov_sig_flat = np.insert(innov_sig_flat, 0, 1, axis=1)
            #innov_sig = ta.sig_to_tensor(innov_sig_flat, 2, order)
            strag[1:, nsim] = np.array([F.dot(ta.sig_to_tensor(innov_sig_flat[it,:], 2, order)) for it in range(env.Nt)])
        return strag
    return strategy