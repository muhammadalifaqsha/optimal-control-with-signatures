from random import random
import numpy as np
from scipy.interpolate import CubicSpline
from tqdm import tqdm
from math import factorial
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
    def __init__(self, b0 = 1, b1 = 1e-3, b2 = 1, sigma0 = 1, sigma2 = 0, X0 = 10, A = 0, B = 1e-1, C = 0, D = 0, E = 1, G = 0, T = 1, Nt = 1_000):
        self.b0 = b0
        self.b0_tilde = self.b0 + 0.5*sigma2*sigma0
        self.b1 = b1
        self.b2 = b2
        self.b2_tilde = self.b2 + 0.5*sigma2**2
        self.sigma0 = sigma0
        self.sigma2 = 0
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
        #self.W_sig = np.zeros(((self.dim**(order+1) - self.dim)//(self.dim - 1), nsims))
        self.X_sig = np.zeros(((self.dim**(order+1) - self.dim)//(self.dim - 1), nsims))
        self.nu = np.zeros((self.Nt+1, nsims))
        self.X[0,:] = self.X0
        self.run_crit = np.zeros((self.Nt+1, nsims))
        for it in range(self.Nt):            
            X_t = self.X[it,:]
            self.X_t = X_t
            dW = self.W[it+1,:] - self.W[it, :]
            nu_t = strategy(self, self.dim, it, nsims)
            self.nu[it, :] = nu_t

            integrand = self.A*X_t**2 + self.B*nu_t**2 + 2*self.C*X_t + 2*self.D*nu_t
            self.run_crit[it+1,:] = self.run_crit[it,:] + integrand*dt
            
            self.X[it+1,:] = self.X[it,:] + (self.b0_tilde + self.b1*nu_t + self.b2_tilde*X_t)*dt + (self.sigma0 + self.sigma2*X_t)*dW

            if it == 0:
                continue
                
            dX = self.X[it,:] - self.X[it-1, :]
            for nsim in range(nsims):
                path_increment = np.array([self.timesteps[it+1]-self.timesteps[it], dX[nsim]])
                #self.W_sig[:, nsim] = iisignature.sigjoin(self.W_sig[:, nsim], path_increment, order)
                self.X_sig[:, nsim] = iisignature.sigjoin(self.X_sig[:, nsim], path_increment, order)
         
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

def sig_strag_dot_state(F, order = 3):
    def strategy(env, dim, it, nsims):
        strag = np.zeros((nsims,))
        for nsim in range(nsims):
            X_sig_flat = np.insert(env.X_sig[:, nsim], 0, 1)
            X_sig = ta.sig_to_tensor(X_sig_flat, dim, order)
            strag[nsim] = F.dot(X_sig)
        return strag
    return strategy

def sig_strag_dot_2(F, order = 3):
    def strategy(env, W_sig, dim, it):
        W_sig_flat = np.insert(W_sig[:], 0, 1)
        W_sig = ta.sig_to_tensor(W_sig_flat, dim, order)
        strag = F.dot(W_sig)
        return strag
    return strategy

def strategy_hjb_sig_functional(env, poly_deg, max_order):
    f2_terminal = np.array([env.E])
    df2 = lambda t, g: -env.b1**2*g**2/env.B + (env.sigma2**2+2*env.b2_tilde)*g + env.A
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

    e0 = ta.word((), env.dim)
    e1 = ta.word((0,), env.dim)
    e2 = ta.word((1,), env.dim)

    coef_0 = - (2*env.b1*f1-2*env.D)/(2*env.B)
    coef_0 = coef_0.reshape((-1,))
    p_coef_0 = np.polyfit(env.timesteps, coef_0, deg = poly_deg)
    coef_x = -(2*env.b1*f2)/(2*env.B)
    coef_x = coef_x.reshape((-1,))
    p_coef_x = np.polyfit(env.timesteps, coef_x, deg = poly_deg)
    factorials = np.array([factorial(i) for i in range(poly_deg + 1)])
    p_coef_0 = p_coef_0[::-1]*factorials
    p_coef_x = p_coef_x[::-1]*factorials

    coef_0_sig = ta.sig_to_tensor([0], dim = env.dim, order = 0)
    coef_x_sig = ta.sig_to_tensor([0], dim = env.dim, order = 0)
    word = tuple()
    for k in range(poly_deg + 1):
        word_tensor = ta.word(word, env.dim)
        coef_0_sig  += word_tensor*p_coef_0[k]
        coef_x_sig  += word_tensor*p_coef_x[k]
        word = (0,) + word

    F = coef_0_sig + coef_x_sig*env.X0 + coef_x_sig.shuffle(e2, max_order = poly_deg + 1)

    F = F.truncate(max_order)
    
    return F

def fawcett(env, max_order, scaled = False):
    T = env.T
    if scaled:
        scale = env.sigma0
    else:
        scale = 1
    exp = ta.Tensor(env.dim, 0)
    e1 = ta.word((0,), env.dim)
    l = e1*env.T
    term = ta.word((), env.dim)
    for d in range(env.dim-1):
        l += (ta.word((d+1, d+1), env.dim)*(env.T/2)*scale**2)
    for n in range(max_order+1):
        exp += term
        term = (l*term)*(1/(n+1))
        term = term.truncate(max_order)
    exp = exp.truncate(max_order)
    return exp
        

def strategy_hjb_feedback(env, dim, it, nsims):
    f2_terminal = np.array([env.E])
    df2 = lambda t, g: -env.b1**2*g**2/env.B + (env.sigma2**2+2*env.b2_tilde)*g + env.A
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

    strag = (-2*env.b1*f2[0,it]*env.X[it,:] - 2*env.b1*f1[0,it]-2*env.D)/(2*env.B)
    
    return strag
    
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