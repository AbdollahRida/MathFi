# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 02:55:23 2018

@author: mohamed.abdel-wedoud
"""

import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt

from tick.plot import plot_hawkes_kernels

from tick.hawkes import HawkesConditionalLaw

import seaborn as sns; sns.set()


#estimateur des kernel
def events(fichier):
    
    
    df = pd.read_csv(fichier)
    
    times = np.array(df["Time"])
    order_side = np.array(df["Side"])
    order_type = np.array(df["OrderType"])
    mid_prices = (np.array(df["AskPriceAfter"]) + np.array(df["BidPriceAfter"])) / 2.
    
    Indicatrices_T_plus = (order_side == 1.) * (order_type == 0.)
    Indicatrices_T_moins = (order_side == - 1.) * (order_type == 0.)
    
    Indicatrices_N_plus = mid_prices[1:] > mid_prices[:-1]
    Indicatrices_N_moins = mid_prices[1:] < mid_prices[:-1]
    
    times = times - times[0]
    
    times_T_plus = times*Indicatrices_T_plus
    times_T_plus = times_T_plus[times_T_plus > 0.]
    
    times_T_moins = times*Indicatrices_T_moins
    times_T_moins = times_T_moins[times_T_moins > 0.]
    
    times_N_plus = times[1:]*Indicatrices_N_plus
    times_N_plus = times_N_plus[times_N_plus > 0.]
    
    times_N_moins = times[1:]*Indicatrices_N_moins
    times_N_moins = times_N_moins[times_N_moins > 0.]
    
    times=[times_T_plus,times_T_moins,times_N_plus,times_N_moins]
    return times


def no_param_estim(fichier) :
    times = events(fichier)
    hawkes_learner = HawkesConditionalLaw(claw_method="log", delta_lag=0.1, min_lag=5e-4, max_lag=500,quad_method="log", n_quad=10, min_support=1e-4, max_support=1, n_threads=2)
    estimate = hawkes_learner.fit([times])
    return estimate


mon_dossier = 'data_projet511'
    


def no_param_estim_n_r(dossier,n_r) :
    
    fichiers = os.listdir(dossier)
    Events = []
    
    my_choices = np.random.choice(len(fichiers),n_r,replace=False)
    k = 0
    
    for i in fichiers :
        if k in my_choices :
            path = dossier + '/'
            path = path + i
            Events.append(events(path))
        k = k + 1
        
    hawkes_learner = HawkesConditionalLaw(claw_method="log", delta_lag=0.1, min_lag=5e-4, max_lag=500,quad_method="log", n_quad=10, min_support=1e-4, max_support=1, n_threads=2)
    estimate = hawkes_learner.fit(Events)
    
    return estimate

hawkes_learner=no_param_estim_n_r(mon_dossier,40)

times = np.linspace(0., hawkes_learner.max_support, 1000)
kernels = []


for i in range(hawkes_learner.n_nodes):
    kernels += [[]]
    for j in range(hawkes_learner.n_nodes):
        kernel_values = hawkes_learner.get_kernel_values(i, j, times)
        kernel_values = np.nan_to_num(kernel_values)
        kernels[-1] += [kernel_values]
        
Phi_T_s = [kernels[0][0], kernels[1][1]] 
Phi_T_c = [kernels[0][1], kernels[1][0]]


Phi_F_s = [ kernels[0][2] , kernels[1][3] ]
Phi_F_c = [ kernels[0][3] , kernels[1][2] ]


Phi_I_s = [ kernels[2][0] , kernels[2][1] ]
Phi_I_c = [ kernels[3][0] , kernels[2][1] ]


Phi_N_s = [ kernels[2][2] , kernels[3][3] ]
Phi_N_c = [ kernels[2][3] , kernels[3][2] ]

def correlation(X) :
    
    return np.sum(np.array(X[0]) * np.array(X[1])) / np.sqrt( (np.sum(np.array(X[0])**2)) * (np.sum(np.array(X[1])**2)))


matrix = np.array([[correlation(Phi_T_s),correlation(Phi_T_c),correlation(Phi_F_s),correlation(Phi_F_c)],[correlation(Phi_T_c),correlation(Phi_T_s),correlation(Phi_F_c),correlation(Phi_F_s)],[correlation(Phi_I_s),correlation(Phi_I_c),correlation(Phi_N_s),correlation(Phi_N_c)],[correlation(Phi_I_c),correlation(Phi_I_s),correlation(Phi_N_c),correlation(Phi_N_s)]])

ax = sns.heatmap(matrix)