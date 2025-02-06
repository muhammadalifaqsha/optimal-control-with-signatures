from random import random
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp
import utils as utils

# simulation

class simulation:
    """
    simulate the models & keep all the components
    """
    def __init__(self, env, inf_trader, uninf_trader, broker, W, alpha, nuU, strategy = "our model", mispecify = False, mispecify_scale = 1, nsims = 10_000):
        self.env = env
        self.inf_trader = inf_trader
        self.uninf_trader = uninf_trader
        self.broker = broker
        self.nsims = nsims
        self.W = W
        self.alpha = alpha
        self.nu_U = nuU
        self.mispecify = mispecify
        self.mispecify_scale = mispecify_scale
        self.strategy = strategy

        z = inf_trader.z
        f3 = z[9] / inf_trader.k
        f0 = z[0] / 2 / inf_trader.k
        f2 = inf_trader.c_importance * z[2] / 2 / inf_trader.k
        f1 = z[1] / 2 / inf_trader.k

        dt = env.T/env.Nt

        self.S = np.zeros((env.Nt + 1, nsims))
        self.S[0, :] = env.S0

        self.H_I = np.zeros((1, nsims))
        self.H_B = np.zeros((1, nsims))

        self.V_I = inf_trader.V_I
        
        if strategy=="alternative filter" or strategy=="naive filter":
            self.V_B = broker.V_B_alt
        else:
            self.V_B = broker.V_B

        self.total_value_traded = np.zeros((1, nsims))

        self.Q_B = np.zeros((env.Nt + 1, nsims))
        self.Q_I = np.zeros((env.Nt + 1, nsims))
        self.X_B = np.zeros((env.Nt + 1, nsims))
        self.X_I = np.zeros((env.Nt + 1, nsims))

        self.Q_B[0, :] = broker.Q0
        self.Q_I[0, :] = inf_trader.Q0
        
        if self.mispecify:
            self.Q_I[0, :] = np.random.normal(loc=inf_trader.Q0, scale = self.mispecify_scale, size=nsims)

        self.nu_B = np.zeros((env.Nt + 1, nsims))
        self.nu_I = np.zeros((env.Nt + 1, nsims))
        self.nu_B_hat = np.zeros((env.Nt + 1, nsims))
        
        self.alpha_hat = np.zeros((env.Nt + 1, nsims))

        for i, t in enumerate(env.timesteps[:-1]):

            self.S[i+1,:] = self.S[i,:] + (env.b*self.nu_B[i,:] + self.alpha[i,:])*dt + env.sigma_s*(W[i+1,:]-W[i,:])
            self.Q_I[i+1, :] = self.Q_I[i, :] + self.nu_I[i, :]*dt
            self.X_I[i+1, :] = self.X_I[i, :] - self.nu_I[i, :]*(self.S[i,:] + inf_trader.k*self.nu_I[i, :])*dt

            self.Q_B[i+1, :] = self.Q_B[i, :] + (self.nu_B[i, :] - self.nu_I[i, :] - self.nu_U[i, :])*dt
            self.X_B[i+1, :] = self.X_B[i, :] - self.nu_B[i, :]*(self.S[i,:] + broker.k*self.nu_B[i, :])*dt + self.nu_I[i, :]*(self.S[i,:] + inf_trader.k*self.nu_I[i, :])*dt + self.nu_U[i, :]*(self.S[i,:] + uninf_trader.k*self.nu_U[i, :])*dt

            # Filtering

            ## trader's perspective

            dY = (self.S[i+1,:] - self.S[i,:]) - self.alpha[i,:]*dt

            d_nu_B_hat = inf_trader.theta_B*(inf_trader.mu_B - self.nu_B_hat[i, :])*dt +  (1/(env.sigma_s)**2)*env.b*inf_trader.V_I_eval(t)*(dY - env.b*self.nu_B_hat[i,:]*dt)
            self.nu_B_hat[i+1,:] = self.nu_B_hat[i,:] + d_nu_B_hat 

             # optimal speed of informed trader

            self.nu_I[i+1, :] = inf_trader.informed_speed(t+dt, self.alpha[i+1, :], self.nu_B_hat[i+1, :], self.Q_I[i+1, :])

            ## broker's perspective
            
            if strategy=="our model":
                dZ = (self.S[i+1,:] - self.S[i,:]) - env.b*self.nu_B[i,:]*dt    
                d_alpha_hat = -env.kappa_alpha*self.alpha_hat[i,:]*dt + (1/env.sigma_s**2)*(broker.V_B_eval(t) + env.corr*env.sigma_s*env.sigma_alpha)*(dZ - self.alpha_hat[i,:]*dt)
                self.alpha_hat[i+1, :] = self.alpha_hat[i, :] + d_alpha_hat
                ## optimal speed of broker
                self.nu_B[i+1, :] = broker.optimal_trading_rate(t+dt, self.Q_B[i+1, :], self.alpha_hat[i+1, :], 
                                                     self.nu_U[i+1, :], self.Q_I[i+1, :] - self.Q_I[0, :])
                
            elif strategy=="alternative filter":
                self.gamma_tilde[i+1, :] = self.nu_I[i+1, :] - f0[i+1] - f3[i+1] * (self.Q_I[i+1, :] - self.Q_I[0, :])
                self.Z_tilde[i+1, :] = self.gamma_tilde[i+1, :]/broker.GF5[i+1]

                if i+1 == env.Nt:
                    self.Z_tilde[i+1, :] = self.Z_tilde[i, :]

                self.integral_subtract[i+1, :] = self.integral_subtract[i, :] + dt*(self.alpha_hat[i,:] * broker.GF7[i])

                self.integral_subtract_Z[i+1, :] = self.integral_subtract_Z[i, :] + dt*(self.Z_tilde[i, :] * broker.GF6[i] + self.gamma_tilde[i, :] * broker.GF0[i]/broker.GF5[i] + self.nu_B[i, :] * broker.GF1[i]/broker.GF5[i] + broker.GF2[i]/broker.GF5[i])
                self.Z[i+1, :] = self.Z_tilde[i+1, :] - self.integral_subtract_Z[i+1, :]
                self.I[i+1, :] = self.Z[i+1, :] - self.integral_subtract[i+1, :]
                self.alpha_hat[i+1, :] = self.alpha_hat[i, :] - env.kappa_alpha*self.alpha_hat[i, :]*dt + (broker.GF7[i]*broker.V_B_alt[i] + env.sigma_alpha * broker.kF[i]) * (self.I[i+1, :] - self.I[i, :])

                if i+1 == env.Nt:
                    self.alpha_hat[i+1, :] = self.alpha_hat[i, :]
                    
                ## optimal speed of broker
                self.nu_B[i+1, :] = broker.optimal_trading_rate(t+dt, self.Q_B[i+1, :], self.alpha_hat[i+1, :], 
                                                     self.nu_U[i+1, :], self.Q_I[i+1, :] - self.Q_I[0, :])

            elif strategy=="naive filter":
                nu_I_test = inf_trader.informed_speed(t+dt, 0, 0, self.Q_I[i+1, :] -self.Q_I[0, :])

                if f1[i+1] != 0:
                    alpha_hat_new_instance = (self.nu_I[i+1, :] - nu_I_test)/f1[i+1]
                else:
                    alpha_hat_new_instance = self.alpha_hat[i, :]

                self.alpha_hat[i+1, :] = alpha_hat_new_instance
                ## optimal speed of broker
                self.nu_B[i+1, :] = broker.optimal_trading_rate(t+dt, self.Q_B[i+1, :], self.alpha_hat[i+1, :], 
                                                     self.nu_U[i+1, :], self.Q_I[i+1, :] - self.Q_I[0, :])
                
            elif strategy=="benchmark 1":
                self.nu_B[i+1, :] = self.nu_I[i+1, :] - self.Q_B[i+1, :]/(env.T-t-dt)
                
            elif strategy=="benchmark 2":
                self.nu_B[i+1, :] = - self.Q_B[i+1, :]/(env.T-t-dt)
                
            elif strategy=="benchmark 3":
                self.nu_B[i+1, :] = self.nu_I[i+1, :] + self.nu_U[i+1, :]

            ## profit function

            dH_I = (inf_trader.rho0 + inf_trader.rho1*self.V_I[i])*(self.Q_I[i,:]**2)*dt
            self.H_I[0, :] -= dH_I

            dH_B = (broker.rho0 + broker.rho1*self.V_B[i])*(self.Q_B[i,:]**2)*dt
            self.H_B[0, :] -= dH_B

            ## value of total volume traded

            dTot = self.S[i,:]*(np.abs(self.nu_B[i, :]) + np.abs(self.nu_I[i, :]) + np.abs(self.nu_U[i, :]))*dt
            self.total_value_traded[0, :] += dTot

        self.H_I[0, :] += self.X_I[-1, :] + self.Q_I[-1, :]*self.S[-1, :] - (inf_trader.beta0 + inf_trader.beta1*self.V_I[-1])*(self.Q_I[-1,:]**2)
        self.H_B[0, :] += self.X_B[-1, :] + self.Q_B[-1, :]*self.S[-1, :] - (broker.beta0 + broker.beta1*self.V_B[-1])*(self.Q_B[-1,:]**2)

        self.Portfolio_I = self.X_I[-1, :] + self.Q_I[-1, :]*self.S[-1, :]
        self.Portfolio_B = self.X_B[-1, :] + self.Q_B[-1, :]*self.S[-1, :]