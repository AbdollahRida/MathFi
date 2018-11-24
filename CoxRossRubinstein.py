# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 14:55:38 2018

@author: Abdollah RIDA

This code contains some helper functions Callt and ArrayProban. I also added an S argument to all functions (making it easier for testing)

Plots and commentaries are in the python notebook. Test functions are in the second .py file.
"""
#Importing libraries

import  numpy               as np                   #Arrays and math
from    scipy.special       import comb             #combinatronics
import  scipy.stats         as si                   #Normal distribution
import  seaborn             as sns                  #Good looking plots
import  matplotlib.pyplot   as plt                  #Plot utilities

#SETTING GLOBAL PARAMETERS FOR PLT

plt.rcParams['figure.figsize'] = (8.0,4.0)
plt.rcParams.update({'font.size':10})
plt.rcParams['xtick.major.pad'] = '5'
plt.rcParams['ytick.major.pad'] = '5'
plt.style.use('ggplot')

#Defining values

S = 100
T = 2
n = 50
r = 0.05
b = 0.05
K = 80
sigma = 0.3

'''
This function is the answer for Q1-a. It returns the array $S^n_j$ of all possible outcomes for the value of the asset at time j.
'''
def Sn(T, n, b, sigma, j, S):

    #Defining h, u and d
    h = T/n
    u = np.exp(b * h + sigma * np.sqrt(h))
    d = np.exp(b * h - sigma * np.sqrt(h))

    #Creating array of zeroes
    Sn = np.zeros(shape = (j+1, 1))

    #loop to fill correct values
    for i in range(j+1):
        Sn[i,0] = S * (u**(j-i)) * (d**i)

    return Sn

'''
This function is the answer for Q1-b. It returns the array of all possible outcomes for the payoff for the European call of the underlying asset. The payoff of the European call is defined as $Max{S^n_n - K, 0}$ where K is the strike of the underlying asset.
'''
def Payoffn(T, n, b, sigma, K, S):

    #Create an array of ones then multiply by K
    Strike = K * np.ones((n+1, 1))

    #Difference then maximum (np.maximum docs)
    return np.maximum(Sn(T, n, b, sigma, n, S) - Strike, 0)

'''
This function is a helper function. It returns the array of payoff multiplied (in an elementwise fashion) by the corresponding probability under the measure $\mathbb{Q}$. This is explained in p.25 of the lectures notes.
'''
def ArrayProban(T, n, r, b, sigma):

    #Defining h, u and d
    h = T/n
    u = np.exp(b * h + sigma * np.sqrt(h))
    d = np.exp(b * h - sigma * np.sqrt(h))

    Rn = np.exp(r * h)
    qn = (Rn - d)/(u - d)

    #Creating array of zeroes
    Qn = np.zeros(shape = (n+1, 1))

    #loop to fill correct values
    for i in range(n+1):
        Qn[i,0] = comb(n, i, exact=True) * ((1 - qn)**(i)) * (qn**(n-i))

    return Qn

'''
This function is the answer for Q1-c. It returns the price of the European call for the underlying asset, with Strike K and price at 0 S at maturity T, as defined in p.25 of the lectures notes.
'''
def Calln(T, n, r, b, sigma, K, S):

    #Dot product of the payoff and probabilities arrays to obtain expectation
    #under probability measure Q
    p = np.vdot(Payoffn(T, n, b, sigma, K, S), ArrayProban(T, n, r, b, sigma))

    #We then discount the result with the interest rate R to obtain the price
    #of the European call
    return np.exp(-r*T) * p

'''
This function is a helper function. It returns the price of the European call for the underlying asset, with Strike K and price at 0 S, at time t.

This is based on a time shift argument:
As defined in p.24 of the lecture notes, the no-arbitrage market price of the contingent claim at t is defined as the expectation operator under probability measure $Q_t-1$ of the the contingent claim. Since this process goes backward in time, we need n-t steps to get to time t.

Thus the no-arbitrage market price of the contingent claim at t is the the no-arbitrage market price of the claim using n-t steps instead of n steps.
'''
def Callt(T, n, r, b, sigma, K, t, S):

    return Calln(T, n-t, r, b, sigma, K, S)

'''
This function is the answer for Q1-d. It returns the array of all possible hedging strategies at time j. This is based on the definitions p.24 of the lectures notes.

To get the array we have to cycle through all the outcomes at time j.
'''
def Deltan(T, n, r, b, sigma, K, j, S):

    #Defining an array of zeros th (for thêta)
    th = np.zeros((j+1, 1))

    #Defining h, u and d
    h = T/n
    u = np.exp(b * h + sigma * np.sqrt(h))
    d = np.exp(b * h - sigma * np.sqrt(h))

    #Sj is the array of possible outcomes at j
    Sj = Sn(T, n, b, sigma, j, S)

    #po is the array of call prices at j+1
    po = [Callt(T, n, r, b, sigma, K, j + 1, Si) for Si in Sn(T, n, b, sigma, j+1, S)]

    #We loop through all possible outcomes to get a strategy for each one
    for each in range(j+1):
        th[each] = (po[each] - po[each + 1]) / (u * Sj[each] - d * Sj[each])

    return th

'''
This function is the answer for Q2-a. It returns the Black_Scholes price at time 0 of the European call option.

This is based on the Theorem 2.8 of p.26 of the lecture notes.
'''
def Call(T, r, sigma, K, S):

    #defining the d+ and d- functions
    dp = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    dm = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    #The call price is therefore
    call = (S * si.norm.cdf(dp, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(dm, 0.0, 1.0))

    return call

'''
This function is the answer for Q2-b. It returns the relative error between the Black_Scholes price at time 0 of the European call option and the Cox-Ross-Rubinstein one.
'''
def err(T, n, r, b, sigma, K, S):

    cox = Calln(T, n, r, b, sigma, K, S)
    bsm = Call(T, r, sigma, K, S)

    return cox/bsm - 1