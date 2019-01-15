"""
Lun. 12 Sep 2018

Author: Abdollah RIDA
"""

#IMPORTING LIBS

import CPS7
import numpy as np 
import scipy.stats as si 
import matplotlib.pyplot as plt 

#VALUES

r = .02
sigma = .4
S0 = 100
T = .9
k_list = [80+i for i in range(41)]
epsilon_list = [1, .5, .1, .01, .001, .0001]

#TESTS

#Test C0 MC:

def test_1(M, n):

    for each in k_list:
        MC = CPS7.MC_C(r, T, S0, each, sigma, M, n)
        Ex = CPS7.C0(r, T, S0, each, sigma)
        print("The Monte Carlo approximation is: {}".format(MC))
        print("The Exact value is: {}".format(Ex))
        print("The error is: {}\n".format(MC - Ex))

#test_1(10000, 365)

#Test Q2

def test_2(M, n, indice_K_a_choisir):
    #indice_K_a_choisir est l'indice de la valeur de K qui nous interesse, entre 0 et 40
    K = k_list[indice_K_a_choisir]
    values = []

    for each in epsilon_list:
        values += [CPS7.Diff_Delta0(r, T, S0, K, sigma, M, n, each)]
    
    Delta_0 = CPS7.Delta0(r, T, S0, K, sigma)
    exact = [Delta_0 for i in range(len(epsilon_list))]

    plt.plot(epsilon_list, values)
    plt.plot(epsilon_list, exact)
    plt.show()

test_2(10000, 365, 0)

epsilon = .1

#Test Q3
#Etant donné la bonne valeur de epsilon, on prend epsilon = .1

def test_3(M, n, epsilon):
    for each in k_list:
        MC = CPS7.Diff_Delta0(r, T, S0, each, sigma, M, n, epsilon)
        Ex = CPS7.MC_Delta(r, T, S0, each, sigma, M, n)
        print("The gradient approximation is: {}".format(MC))
        print("The representation value is: {}".format(Ex))
        print("The error is: {}\n".format(MC - Ex))

#test_3(10000, 365, epsilon)