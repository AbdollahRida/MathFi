#!/usr/bin/env python
# Black-Scholes PDE solving using DGM paper
# __author__ = "Abdollah Rida"
# __email__ = "abdollah.rida@berkeley.edu"

# Import needed packages

import numpy as np
import scipy.stats as spstats

from __params__ import *

# Black-Scholes European call price

# Analytical known solution
def lambd_H(t):
    '''
    lambda_H term for EU Call price under fBS

    Args:
    ----
        t:     time
    '''
    global H

    return 2*H*t**(2*H - 1)

def dp(S, K, r, sigma, t):
    global H

    log = np.log(S/K)
    num = (r + lambd_H(t)/2 * sigma**2) * (T - t)
    denom = sigma * np.sqrt(lambd_H(t) * (T - t))
    return (log + num)/denom

def dm(S, K, r, sigma, t):
    global H

    log = np.log(S/K)
    num = (r - lambd_H(t)/2 * sigma**2) * (T - t)
    denom = sigma * np.sqrt(lambd_H(t) * (T - t))
    return (log + num)/denom

def BlackScholesCall(S, K, r, sigma, t):
    '''
    Analytical solution for European call option price under
    Black-Scholes model

    Args:
    ----
        S:     spot price
        K:     strike price
        r:     risk-free interest rate
        sigma: volatility
        t:     time
    '''
    global H

    # first term
    ft = S * spstats.norm.cdf(dp(S, K, r, sigma,t))

    # second term
    st = K * np.exp(-r * (T-t)) * spstats.norm.cdf(dm(S, K, r, sigma,t))

    callPrice = ft - st

    return callPrice
