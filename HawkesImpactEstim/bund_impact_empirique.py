# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 16:12:41 2018

@author: mohamed.abdel-wedoud
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 15:54:45 2018

@author: mohamed.abdel-wedoud
"""

"""
==========================
Fit Hawkes on finance data
==========================
This example fit hawkes kernels on finance data provided by `tick-datasets`_. 
repository.
Kernels norms of a Hawkes process fit on finance data of Bund market place.
This reproduces experiments run in
Bacry, E., Jaisson, T. and Muzy, J.F., 2016. Estimation of slowly decreasing
Hawkes kernels: application to high-frequency order book dynamics.
Quantitative Finance, 16(8), `pp.1179-1201`_.
with  :math:`P_u` (resp. :math:`P_d`) counts the number of upward (resp.
downward) mid-price moves and :math:`T_A` (resp. :math:`T_b`) counts the
number of market  orders at the ask (resp. bid) that do not move the price.
We observe expected  behavior with for example mid-price moving downward
triggering (resp. preventing) market orders at the ask (resp. at the bid).
.. _tick-datasets: https://github.com/X-DataInitiative/tick-datasets
.. _pp.1179-1201: http://www.tandfonline.com/doi/abs/10.1080/14697688.2015.1123287
"""
import numpy as np

from tick.dataset import fetch_hawkes_bund_data
from tick.hawkes import HawkesConditionalLaw
from tick.plot import plot_hawkes_kernel_norms

from tick.plot import plot_hawkes_kernels

from tick.base import TimeFunction

import seaborn as sns; sns.set()


import matplotlib.pyplot as plt

timestamps_list = fetch_hawkes_bund_data()
abscisses = np.linspace(0.,10.,200)

def R_X_realisation(element):
    
    times = np.array(element)
    
    times_N_plus = times[2]
    
    times_N_moins = times[3]
    
    time_axe = np.sort(np.concatenate((times_N_moins,times_N_plus)))
    
    I_R_X =[]
    
    for i in time_axe:
        if i in times_N_moins :
            I_R_X.append(-1.)
        else : I_R_X.append(1)
            
    
    R_X = np.cumsum(I_R_X)

    
    k = 0
    
    valeurs_R_X = []
    
    for i in range(200) :
        
        while abscisses[i] > time_axe[k] : k = k+1 
        
        valeurs_R_X.append(R_X[k])
        
    return np.array(valeurs_R_X)


R = np.zeros(200)
k=0.

for i in timestamps_list :
    if i[0][0] == 0. :
        R = R + R_X_realisation(i)
        k=k+1.

R_em = R / k

print("taille d'echantillon est {}".format(int(k)))

plt.figure()

plt.plot(abscisses,R_em)

plt.show()

