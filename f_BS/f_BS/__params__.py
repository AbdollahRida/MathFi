# Imports
import time

# Parameters

# Option parameters
r = 0.05           # Interest rate
sigma = 0.05       # Volatility
K = 100            # Strike
H = 0.5           # Hurst parameter
T = 1              # Terminal time
S0 = 100           # Initial price

# PDE solving domain
t_low = 0 + 1e-10    # time lower bound
S_low = 0.0 + 1e-10  # spot price lower bound
S_high = 2*K         # spot price upper bound

# neural network parameters
num_layers = 5
nodes_per_layer = 50
learning_rate = 0.001

# Training parameters
sampling_stages  = 200   # n of times to resample new time-space domain
steps_per_sample = 10    # n of SGD steps to take before re-sampling

# Sampling parameters
nSim_interior = 1000
nSim_terminal = 100
S_multiplier  = 1.5   # multiplier for oversampling

# time values at which to examine density
# Just 4 for now, will change
valueTimes = [t_low, T/3, 2*T/3, T]

# Plot options
n_plot = 41  # Points on plot grid for each dimension

# Save options
date = time.strftime("%d-%m-%Y--%H-%M")

saveOutput = True
saveName   = 'f_BlackScholes_EuropeanCall_Hurst_' + str(H) + '_' + date
saveFigure = True
figureName = 'f_BlackScholes_EuropeanCall_Hurst_' + str(H) + '_' + date + '.png'
