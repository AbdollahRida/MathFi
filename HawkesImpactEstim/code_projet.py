# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 19:54:50 2018

@author: mohamed.abdel-wedoud
"""

import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt


from tick.plot import plot_hawkes_kernels

from tick.hawkes import  HawkesSumGaussians,HawkesSumExpKern

from tick.hawkes import HawkesConditionalLaw 





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


#hawkes_learner=no_param_estim("xFDAX_20130701.csv")
#
#print(hawkes_learner.baseline)

#plot_hawkes_kernels(hawkes_learner)
#
#plot_hawkes_kernel_norms(hawkes_learner,node_names=["P_u", "P_d", "T_a", "T_b"])

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
    
hawkes_learner=no_param_estim_n_r(mon_dossier,100)

#print(hawkes_learner.baseline)

plot_hawkes_kernels(hawkes_learner)



#plot_hawkes_kernel_norms(hawkes_learner,node_names=["P_u", "P_d", "T_a", "T_b"])





times = np.linspace(0., hawkes_learner.max_support, 1000)
kernels = []


for i in range(hawkes_learner.n_nodes):
    kernels += [[]]
    for j in range(hawkes_learner.n_nodes):
        kernel_values = hawkes_learner.get_kernel_values(i, j, times)
        kernel_values = np.nan_to_num(kernel_values)
        kernels[-1] += [kernel_values]



Nabla = hawkes_learner.mean_intensity

Nabla_N = (Nabla[2] + Nabla[3]) / 2.
Nabla_T = (Nabla[0] + Nabla[1]) / 2.

Phi_T_s = ( kernels[0][0] + kernels[1][1] ) / 2.
Phi_T_c = ( kernels[0][1] + kernels[1][0] ) / 2.
Delta_Phi_T = Phi_T_s - Phi_T_c

Phi_F_s = ( kernels[0][2] + kernels[1][3] ) /2.
Phi_F_c = ( kernels[0][3] + kernels[1][2] ) / 2.
Delta_Phi_F = Phi_F_s - Phi_F_c

Phi_I_s = ( kernels[2][0] + kernels[2][1] ) / 2.
Phi_I_c = ( kernels[3][0] + kernels[2][1] ) / 2.
Delta_Phi_I = Phi_I_s - Phi_I_c

Phi_N_s = ( kernels[2][2] + kernels[3][3] ) / 2.
Phi_N_c = ( kernels[2][3] + kernels[3][2] ) / 2.
Delta_Phi_N = Phi_N_s - Phi_N_c


z_axe = np.arange(-10,10,0.05)


def TF(Phi,z,times1) :
    
    return np.sum(np.exp( - complex(0,z) * times1) * Phi) * (times1[1]-times1[0])


def ITF(Phi_tilda,z_axe1,t) :
    
    return np.sum(np.exp( complex(0,t) * z_axe1) * Phi_tilda) * (z_axe1[1]-z_axe1[0])


Delta_Phi_T_tilda = np.array([TF(Delta_Phi_T,i,times) for i in z_axe])
Delta_Phi_F_tilda = np.array([TF(Delta_Phi_F,i,times) for i in z_axe])
Delta_Phi_N_tilda = np.array([TF(Delta_Phi_N,i,times) for i in z_axe])
Delta_Phi_I_tilda = np.array([TF(Delta_Phi_I,i,times) for i in z_axe])


numerateur_r_tilda = (1. - Delta_Phi_N_tilda.conjugate())*Delta_Phi_I_tilda + (Nabla_N/Nabla_T) * (1. - Delta_Phi_T_tilda) * Delta_Phi_F_tilda.conjugate()
terme_denominateur_r_tilda = (1. - Delta_Phi_N_tilda) * (1. - Delta_Phi_T_tilda) - Delta_Phi_I_tilda * Delta_Phi_F_tilda
denominateur_r_tilda  = terme_denominateur_r_tilda * terme_denominateur_r_tilda.conjugate()


I = np.trapz(Phi_I_s,times)

times = np.linspace(0., 2 * hawkes_learner.max_support, 1000)

r_tilda = numerateur_r_tilda / denominateur_r_tilda
    
r = np.array([ ITF(r_tilda,z_axe,i) for i in times])

r = r.real

R = np.cumsum(r[:-1] + r[1:]) * (times[1]-times[0]) / 2.

plt.figure()
plt.plot(times[1:], R , label ='market impact evolution')
plt.legend()
plt.show()


#
#numerateur_ep_tilda = (1. - Delta_Phi_N_tilda.conjugate()) + (Nabla_N/Nabla_T) * (1. - Delta_Phi_T_tilda) * Delta_Phi_F_tilda.conjugate() *(1/I)
#terme_denominateur_ep_tilda = (1. - Delta_Phi_N_tilda) * (1. - Delta_Phi_T_tilda) - I * Delta_Phi_F_tilda
#denominateur_ep_tilda  = terme_denominateur_ep_tilda * terme_denominateur_ep_tilda.conjugate()
#
#epsilon_tilda = 1 - (numerateur_ep_tilda / denominateur_ep_tilda )
#
#epsilon = np.array([ ITF(epsilon_tilda,z_axe,i) for i in times])
#
#
#epsilon = epsilon.real
#
#R = I * ( 1. - (np.cumsum(epsilon[:-1] + epsilon[1:]) * (times[1]-times[0]) / 2. ))
#
#plt.figure()
#plt.plot(times[1:], R , label ='market impact evolution')
#plt.legend()
#plt.show()

