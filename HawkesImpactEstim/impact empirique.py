# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 23:41:18 2018

@author: mohamed.abdel-wedoud
"""

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

from tick.hawkes import SimuHawkes, HawkesKernelExp, HawkesKernelTimeFunc

from tick.hawkes import HawkesConditionalLaw

from tick.base import TimeFunction




abscisses = np.linspace(0.,10.,200)

def R_X_realisation(fichier):
    
    df = pd.read_csv(fichier)
    
    times = np.array(df["Time"])

    mid_prices = (np.array(df["AskPriceAfter"]) + np.array(df["BidPriceAfter"])) / 2.
    
    
    Indicatrices_N_plus = mid_prices[1:] > mid_prices[:-1]
    Indicatrices_N_moins = mid_prices[1:] < mid_prices[:-1]
    
    R_X = np.cumsum(Indicatrices_N_plus) - np.cumsum(Indicatrices_N_moins )


    times_N_plus = times[1:]*Indicatrices_N_plus
    times_N_plus = times_N_plus[times_N_plus > 0.]
    
    times_N_moins = times[1:]*Indicatrices_N_moins
    times_N_moins = times_N_moins[times_N_moins > 0.]
    
    time_axe = np.sort(np.concatenate((times_N_moins,times_N_plus))) - times[0]
    
    k = 0
    
    valeurs_R_X = []
    
    for i in range(200) :
        
        while abscisses[i] > time_axe[k] : k = k+1 
        
        valeurs_R_X.append(R_X[k])
 
    
    return np.array(valeurs_R_X)





mon_dossier = 'data_projet511'
    


def R_empirique(dossier) :
    
    fichiers = os.listdir(dossier)
    
    R = np.zeros(200)
    k = 0.
    
    for i in fichiers :
        df = pd.read_csv(dossier + '/' + i) 
        if  ( 1 == (np.array(df["Side"]))[0])  and ( 0 == (np.array(df["OrderType"]))[0]) :
            R = R + R_X_realisation(dossier + '/' + i)
            k = k + 1.
        
    if k > 0. : 
        R = R / k
        
    else :
        return np.zeros(200)
    
    return R , k
    

R_em, k  = R_empirique(mon_dossier)

print("taille d'echantillon est {}".format(int(k)))

plt.figure()

plt.plot(abscisses,R_em)

plt.show()