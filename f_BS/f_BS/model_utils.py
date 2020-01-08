#!/usr/bin/env python
# Black-Scholes PDE solving using DGM paper
# __author__ = "Abdollah Rida"
# __email__ = "abdollah.rida@berkeley.edu"

# Import needed packages

import DGM
import tensorflow as tf
import numpy as np

from __params__ import *

# Sampling function - randomly sample time-space pairs

def sampler(nSim_interior, nSim_terminal):
    '''
    Sample time-space points from the function's domain; points are
    sampled uniformly on the interior of the domain, at the
    initial/terminal time points and along the spatial boundary at
    different time points.

    Args:
    ----
        nSim_interior: number of space points in the interior of the function's domain to sample
        nSim_terminal: number of space points at terminal time to sample (terminal condition)
    '''

    # Sampler #1: domain interior
    t_interior = np.random.uniform(low=t_low, high=T, size=[nSim_interior, 1])
    S_interior = np.random.uniform(low=S_low, high=S_high*S_multiplier, size=[nSim_interior, 1])

    # Sampler #2: spatial boundary
        # no spatial boundary condition for this problem

    # Sampler #3: initial/terminal condition
    t_terminal = T * np.ones((nSim_terminal, 1))
    S_terminal = np.random.uniform(low=S_low, high=S_high*S_multiplier, size = [nSim_terminal, 1])

    return t_interior, S_interior, t_terminal, S_terminal

# Loss function for Fokker-Planck equation

def loss(model, t_interior, S_interior, t_terminal, S_terminal):
    '''
    Compute total loss for training.

    Args:
    ----
        model:      DGM model object
        t_interior: sampled time points in the interior of the fct
        domain
        S_interior: sampled space points in the interior of the fct
        domain
        t_terminal: sampled time points at terminal point
        (vector of terminal times)
        S_terminal: sampled space points at terminal time
    '''
    global H
    global K
    global sigma
    global r

    # Loss term #1: PDE
    # compute function value and derivatives at current sampled points
    V = model(t_interior, S_interior)
    V_t = tf.gradients(V, t_interior)[0]
    V_s = tf.gradients(V, S_interior)[0]
    V_ss = tf.gradients(V_s, S_interior)[0]
    diff_V = V_t + H * t_interior**(2*H -1) * sigma**2 * S_interior**2 * V_ss + r * S_interior * V_s - r*V

    # compute average L2-norm of differential operator
    L1 = tf.reduce_mean(tf.square(diff_V))

    # Loss term #2: boundary condition
        # no boundary condition for this problem

    # Loss term #3: initial/terminal condition
    target_payoff = tf.nn.relu(S_terminal - K)
    fitted_payoff = model(t_terminal, S_terminal)

    L3 = tf.reduce_mean(tf.square(fitted_payoff - target_payoff))

    return L1, L3
