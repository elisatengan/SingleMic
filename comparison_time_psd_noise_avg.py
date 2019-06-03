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
angles_sources, sources_signals = generate_noise(nb_sources, fs, duration_sig, 'uniform',np.array([1,1]))
nsamples_seg, delta_theta, y = generate_mic_output(sources_signals, angles_sources, n_pos, 'cardioid', theta_m=0)


"""
Estimating the PSD of the microphone's output signal and the original sources
"""

winlen = 512
overlap = 0
# overlap = int(winlen/2)

# Total number of overlapping windows in time
nwindows = int((len(y) - overlap)/(winlen - overlap))

psds1 = np.empty((winlen, nwindows))
psds2 = np.empty((winlen, nwindows))
psdsy = np.empty((winlen, nwindows))
tmp1 = np.zeros(shape=(winlen,))
tmp2 = np.zeros(shape=(winlen,))
tmpy = np.zeros(shape=(winlen,))


# PSD estimation for each segment
# It still looks very confusing. I basically used my Welch implementation but with only one windowed segment.
# I think it is equivalent to the Blackman-Tukey method
# TODO: Create another function to make this code look less confusing or just adapt the PSD estimator

for idx in range(nwindows):
    freq_psd1, psds1[:, idx] = estimate_psd(sources_signals[0, idx * (winlen - overlap):idx * (winlen-overlap) + winlen], fs, method='welch', window='boxcar', nperseg=winlen, noverlap=0)
    freq_psd2, psds2[:, idx] = estimate_psd(sources_signals[1, idx * (winlen - overlap):idx * (winlen-overlap) + winlen], fs, method='welch', window='boxcar', nperseg=winlen, noverlap=0)
    freq_psdy, psdsy[:, idx] = estimate_psd(y[idx * (winlen - overlap):idx * (winlen-overlap) + winlen], fs, method='welch', window='boxcar', nperseg=winlen, noverlap=0)


# Number of windows to consider for averaging (Welch)
L = 16

# Exponential averaging
alpha = 0.90  # Exponential averaging factor

numfreq = len(freq_psdy)
psds1_welch = np.empty((numfreq, nwindows))
psds2_welch = np.empty((numfreq, nwindows))
psdsy_welch = np.empty((numfreq, nwindows))

psds1_exp = np.empty((numfreq, nwindows))
psds2_exp = np.empty((numfreq, nwindows))
psdsy_exp = np.empty((numfreq, nwindows))

tmp1 = np.zeros(shape=(numfreq,))
tmp2 = np.zeros(shape=(numfreq,))
tmpy = np.zeros(shape=(numfreq,))


for idx in range(nwindows):

    # Welch averaging
    if idx < L-1:
        psds1_welch[:, idx] = np.sum(psds1[:, 0:idx+1], axis=1)/L
        psds2_welch[:, idx] = np.sum(psds2[:, 0:idx + 1], axis=1) / L
        psdsy_welch[:, idx] = np.sum(psdsy[:, 0:idx + 1], axis=1) / L

    else:
        psds1_welch[:, idx] = np.mean(psds1[:, idx - (L-1):idx+1], axis=1)
        psds2_welch[:, idx] = np.mean(psds2[:, idx - (L - 1):idx + 1], axis=1)
        psdsy_welch[:, idx] = np.mean(psdsy[:, idx - (L - 1):idx + 1], axis=1)

    # Exponential averaging
    tmp1 = alpha*tmp1 + (1-alpha)*(psds1[:, idx])
    tmp2 = alpha*tmp2 + (1-alpha)*(psds2[:, idx])
    tmpy = alpha*tmpy + (1-alpha)*(psdsy[:, idx])

    psds1_exp[:, idx] = tmp1
    psds2_exp[:, idx] = tmp2
    psdsy_exp[:, idx] = tmpy


"""
Cheating output
"""
theta_m=0
delta_theta = 2*math.pi/n_pos

# Auxiliary matrices with angles for creating coefficient matrix
theta_matrix = np.array([np.arange(360)*math.pi/180,]*n_pos)
theta_shifts = np.array([np.arange(n_pos)*delta_theta,]*theta_matrix.shape[1]).transpose()
theta_response = theta_matrix - theta_shifts - theta_m

# Creating vector with sources' PSD for a single frequency. We know that sources are at 0 and 180 degrees
idx_freq = 200
x = np.zeros((360, nwindows))
x[0, :] = psds1_exp[idx_freq, :]
x[180, :] = psds2_exp[idx_freq, :]

# Cheating PSD estimation of sources
matrix_A = np.power(mic_resp['cardioid'](theta_response), 2)
big_matrix_A = np.repeat(matrix_A, (nwindows/n_pos), axis=0)
psdsy_cheat = np.zeros((nwindows,))
for i in range(nwindows):
    psdsy_cheat[i] = big_matrix_A[i, :]@x[:, i]

taxis = np.arange(len(y), step=(winlen - overlap)) / fs
# taxis = taxis[0:len(taxis) - 1]  # used this when there were overlapping windows

y_psd_single_freq = psdsy_exp[idx_freq, :]
fig = plt.figure()
plt.plot(taxis, y_psd_single_freq)
plt.plot(taxis, psdsy_cheat)
plt.legend([r'$\Phi_{Y}$', r'$A@\Phi_{S}$'])
plt.xlabel("Time (s)")
plt.show()
print("Hello World")

