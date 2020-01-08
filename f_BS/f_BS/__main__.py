#!/usr/bin/env python
# Black-Scholes PDE solving using DGM paper
# __author__ = "Abdollah Rida"
# __email__ = "abdollah.rida@berkeley.edu"

# Import needed packages

import DGM
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import f_Call
import model_utils

from __params__ import *

np.seterr(divide='ignore')

#%% Set up network

# initialize DGM model (last input: space dimension = 1)
model = DGM.DGMNet(nodes_per_layer, num_layers, 1)

# tensor placeholders (_tnsr suffix indicates tensors)
# inputs (time, space domain interior, space domain at initial time)
t_interior_tnsr = tf.compat.v1.placeholder(tf.float32, [None,1])
S_interior_tnsr = tf.compat.v1.placeholder(tf.float32, [None,1])
t_terminal_tnsr = tf.compat.v1.placeholder(tf.float32, [None,1])
S_terminal_tnsr = tf.compat.v1.placeholder(tf.float32, [None,1])

# loss
L1_tnsr, L3_tnsr = model_utils.loss(model, t_interior_tnsr, S_interior_tnsr, t_terminal_tnsr, S_terminal_tnsr)
loss_tnsr = L1_tnsr + L3_tnsr

# option value function
V = model(t_interior_tnsr, S_interior_tnsr)

# set optimizer
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_tnsr)

# initialize variables
init_op = tf.compat.v1.global_variables_initializer()

# open session
sess = tf.compat.v1.Session()
sess.run(init_op)

# Train network
# for each sampling stage
for i in range(sampling_stages):

    # sample uniformly from the required regions
    t_interior, S_interior, t_terminal, S_terminal = model_utils.sampler(nSim_interior, nSim_terminal)

    # for a given sample, take the required number of SGD steps
    for _ in range(steps_per_sample):
        loss,L1,L3,_ = sess.run([loss_tnsr, L1_tnsr, L3_tnsr, optimizer],
                                feed_dict = {t_interior_tnsr:t_interior, S_interior_tnsr:S_interior, t_terminal_tnsr:t_terminal, S_terminal_tnsr:S_terminal})

    if i%10 == 0:
        print('Sampling round: {} \n ####### \n Loss so far: {} \n L1 err: {} \n L3 err: {} \n #######'.format(i, loss, L1, L3))

# save outout
if saveOutput:
    saver = tf.train.Saver()
    saver.save(sess, './SavedNets/' + saveName)

# LaTeX rendering for text in plots
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# figure options
plt.figure()
plt.figure(figsize = (12,10))

# vector of t and S values for plotting
S_plot = np.linspace(S_low, S_high, n_plot)

for i, curr_t in enumerate(valueTimes):

    # specify subplot
    plt.subplot(2,2,i+1)

    # simulate process at current t
    optionValue = f_Call.BlackScholesCall(S_plot, K, r, sigma, curr_t,)

    # compute normalized density at all x values to plot and current t value
    t_plot = curr_t * np.ones_like(S_plot.reshape(-1,1))
    fitted_optionValue = sess.run([V], feed_dict= {t_interior_tnsr:t_plot, S_interior_tnsr:S_plot.reshape(-1,1)})

    # plot histogram of simulated process values and overlay estimated density
    plt.plot(S_plot, optionValue, color = 'b', label='Analytical Solution', linewidth = 3, linestyle=':')
    plt.plot(S_plot, fitted_optionValue[0], color = 'r', label='DGM estimate')

    # subplot options
    plt.ylim(ymin=0.0, ymax=K)
    plt.xlim(xmin=0.0, xmax=S_high)
    plt.xlabel(r"Spot Price", fontsize=15, labelpad=10)
    plt.ylabel(r"Option Price", fontsize=15, labelpad=20)
    plt.title(r"\boldmath{$t$}\textbf{ = %.2f}"%(curr_t), fontsize=18, y=1.03)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.grid(linestyle=':')

    if i == 0:
        plt.legend(loc='upper left', prop={'size': 16})

# adjust space between subplots
plt.subplots_adjust(wspace=0.3, hspace=0.4)

#plt.show()

if saveFigure:
    plt.savefig(figureName)
