"""
Lun. 12 Sep 2018

Author: Abdollah RIDA
"""

#IMPORTING LIBS

import numpy as np 
import scipy.stats as si 
import matplotlib.pyplot as plt 

#Helper functions:

def d_m(x, v):
    d_ = np.log(x) / np.sqrt(v) - .5 * np.sqrt(v)
    return d_

#Q1-a

def C0(r, T, S0, K, sigma):

    X0 = S0 / (K * np.exp(-r*T))

    C = si.norm.cdf(- d_m(X0, sigma**2 * T))

    return np.exp(-r*T) * C

def Delta0(r, T, S0, K, sigma):

    X0 = S0 / (K * np.exp(-r*T))

    D = si.norm.pdf(- d_m(X0, sigma**2 * T))

    return - np.exp(-r*T) * D / (S0 * np.sqrt(sigma**2 * T))

#Q1-b

def Brownian(n, i, T):
	#We fist define the time step Dt
    dt = T/n

    #We then create an array of i drawings following N(0, Dt)
    z = np.random.normal(0, 1, i) * np.sqrt(dt)

    #We specify W_0 = 0
    z[0] = 0

    #We then compute the cumulative sums to obtain W
    W = np.cumsum(z)
    return list(W)

def S_(r, sigma, S0, T, n, W):
    S__ = []

    for i in range(n):
    	S__ += [S0 * np.exp((r - sigma**2 / 2) * i * T/n + sigma * W[i])]
    
    return S__

def MC_C(r, T, S0, K, sigma, M, n):
    Ind_ = []

    for k in range(M):
        W = Brownian(n, n, T)
        S_T = S_(r, sigma, S0, T, n, W)[-1]
        if S_T <= K:
            Ind_ += [1]
        else:
            Ind_ += [0]
    
    Ind_ = np.array(Ind_)
    Esp = np.cumsum(Ind_)[-1] / M

    return np.exp(-r * T) * Esp

def Diff_Delta0(r, T, S0, K, sigma, M, n, epsilon):
    return (MC_C(r, T, S0 + epsilon, K, sigma, M, n) - MC_C(r, T, S0 - epsilon, K, sigma, M, n))/ (2 * epsilon)

def MC_Delta(r, T, S0, K, sigma, M, n):
    Ind_ = []

    for k in range(M):
        W = Brownian(n, n, T)
        S_T = S_(r, sigma, S0, T, n, W)[-1]
        if S_T <= K:
            Ind_ += [W[-1] / (S0 * sigma * T)]
        else:
            Ind_ += [0]
    
    Ind_ = np.array(Ind_)
    Esp = np.cumsum(Ind_)[-1] / M

    return np.exp(-r * T) * Esp
    