"""

Elisa Tengan Pires de Souza
KU Leuven
Department of Electrical Engineering (ESAT)

E-mail: elisa.tengan@esat.kuleuven.be

Stadius Center for Dynamical Systems, Signal Processing and Data Analytics (STADIUS)
Kasteelpark Arenberg 10
3001 Leuven (Heverlee)
Belgium



May 2019


"""
import numpy as np
import matplotlib.pyplot as plt
from signal_generator import *
from psd_estimator import *
from noise_generator import *


print("Hello comparison_time_psd_noisever.py")

"""
Signal generation
"""

# Setting parameters
fs = 16000
nb_sources = 2
duration_sig = 6*((2**14)/16000)  # I am cheating in order to get frames with size as a power of 2
Nsamples_total = int(round((duration_sig*fs)))  # used only for debugging
n_pos = 6

# Generating signals
angles_sources, sources_signals = generate_noise(nb_sources, fs, duration_sig, positions='uniform',
                                                 noise_power=np.array([1,1]))
nsamples_seg, delta_theta, y = generate_mic_output(sources_signals, angles_sources, n_pos, 'cardioid', theta_m=0)

"""
Estimating the PSD of the microphone's output signal
"""
welch_nperseg = 512
welch_overlap = 256
y_psd_hat_periodogram = np.empty((welch_nperseg,n_pos))
for i in range(n_pos):
    seg = y[i*nsamples_seg:(i+1)*nsamples_seg]
    freq, y_psd_hat_periodogram[:, i] = estimate_psd(seg, fs, method='welch', nperseg=welch_nperseg, noverlap=welch_overlap)


"""
Cheating output
"""
theta_m=0
delta_theta = 2*math.pi/n_pos

# Auxiliary matrices with angles for creating coefficient matrix
theta_matrix = np.array([np.arange(360)*math.pi/180,]*n_pos)
theta_shifts = np.array([np.arange(n_pos)*delta_theta,]*theta_matrix.shape[1]).transpose()
theta_response = theta_matrix - theta_shifts - theta_m

# Cheating PSD estimation of sources
welch_nperseg = 512
welch_overlap = 256
psd_S = np.empty((welch_nperseg,nb_sources))
for i in range(nb_sources):
    freq, psd_S[:, i] = estimate_psd(sources_signals[i, 0:nsamples_seg], fs, method='welch', nperseg=welch_nperseg,
                                     noverlap=welch_overlap)


# Creating vector with sources' PSD for a single frequency. We know that sources are at 0 and 180 degrees
idx_freq = 200
x = np.zeros((360,))
x[0] = psd_S[idx_freq, 0]
x[180] = psd_S[idx_freq, 1]

# Finally, cheating PSD estimation of microphone output
matrix_A = np.power(mic_resp['cardioid'](theta_response), 2)
y_psd_cheat_singlefreq = matrix_A @ x
y_psd_periodogram_singlefreq = y_psd_hat_periodogram[idx_freq, :]

# Plotting comparison between "feasible" PSD estimation and cheated version
fig = plt.figure()
plt.plot(y_psd_periodogram_singlefreq)
plt.plot(y_psd_cheat_singlefreq)
plt.legend([r'$\Phi_{Y}$', r'$A@\Phi_{S}$'])
plt.xlabel("Position")
plt.xticks(ticks=np.arange(6), labels=[r'$0\degree$', r'$60\degree$', r'$120\degree$', r'$180\degree$', r'$240\degree$', r'$300\degree$'])
plt.show()
print("Hello World")

