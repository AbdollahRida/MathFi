# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 14:55:38 2018

@author: Abdollah RIDA

This code contains the test functions. Make sure that the CoxRossRubinstein.py file is in the same directory as this one.
"""
#Importing libraries

from   CoxRossRubinstein    import 	Calln, Deltan, err
import seaborn 				as 		sns
import matplotlib.pyplot 	as 		plt

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
The following code is the answer for Q1-e.

Let's examine the dependence effect of the strike $K$ on the functions ``Calln`` and ``Deltan``.
'''

call = [Calln(T, n, r, b, sigma, K + i, S) for i in range(41)]
sns.scatterplot([i for i in range(41)], call)
plt.show()

'''
We can see that the call price is a _**decreasing convex**_ function of $K$.
'''

delta = [Deltan(T, n, r, b, sigma, K + i, 0, S) for i in range(41)]
sns.scatterplot([i for i in range(41)], delta)
plt.show()

'''
We can see that the Hedging strategy at 0 is a _**decreasing**_ function of $K$. Since all stock prices at time $j$ can be considered starting stock prices for a new binomial tree, the corresponding hedging strategies can be seen as hedging strategies at 0 for these trees.

We thus obtain that the hedging strategy at time $j$ is a _**decreasing**_ function of $K$.
'''

'''
The following code is the answer for Q2-b.

Let's examine the dependence effect of the strike $K$ on the functions ``Calln`` and ``Deltan``.
'''

#Redefining values

S = 100
T = 2
r = 0.05
b = 0.05
K = 105
sigma = 0.3
N = 201

call = [err(T, i, r, b, sigma, K, S) for i in range(1, N)]
sns.scatterplot([i for i in range(1, N)], call)
plt.show()

'''
We can see that the Cox-Ross-Rubinstein model converges relatively quickly (the error nears 0). And for n values superior to 100 the error is nearly zero.
'''