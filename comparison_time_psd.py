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


print("Hello comparison_time_psd.py")

"""
Signal generation
"""

# Setting parameters
fs = 16000
f0 = 200
nb_sources = 2
duration_sig = 6
n_pos = 6

# Generating signals
angles_sources, sources_signals = generate_sources(nb_sources, fs, f0, duration_sig, 'uniform')
nsamples_seg, delta_theta, y = generate_mic_output(sources_signals, angles_sources, n_pos, 'cardioid', theta_m=0)

"""
Estimating the PSD of the microphone's output signal
"""
y_psd_hat_periodogram = np.empty((nsamples_seg, n_pos))
for i in range(n_pos):
    seg = y[i*nsamples_seg:(i+1)*nsamples_seg]
    freq, y_psd_hat_periodogram[:, i] = estimate_psd(seg, fs, 'periodogram')


"""
Cheating output
"""
theta_m = 0
delta_theta = 2*math.pi/n_pos

# Auxiliary matrices with angles for creating coefficient matrix
theta_matrix = np.array([np.arange(360)*math.pi/180, ]*n_pos)
theta_shifts = np.array([np.arange(n_pos)*delta_theta, ]*theta_matrix.shape[1]).transpose()
theta_response = theta_matrix - theta_shifts - theta_m

# Cheating PSD estimation of sources
psd_S = np.empty((nsamples_seg, nb_sources))

for i in range(nb_sources):
    freq, psd_S[:, i] = estimate_psd(sources_signals[i, 0:nsamples_seg], fs, method='periodogram')

# Creating vector with sources' PSD for f = f0 = 200Hz. We know that sources are at 0 and 180 degrees
x = np.zeros((360,))
x[0] = psd_S[200, 0]
x[180] = psd_S[200, 1]

# Finally, cheating PSD estimation of microphone output
matrix_A = np.power(mic_resp['cardioid'](theta_response), 2)
y_psd_cheat = matrix_A @ x
y_psd_periodogram_200 = y_psd_hat_periodogram[200, :]

# Plotting comparison between "feasible" PSD estimation and cheated version
fig = plt.figure()
plt.plot(y_psd_periodogram_200)
plt.plot(y_psd_cheat)
plt.legend([r'$\Phi_{Y}(f=%s Hz)$' % f0, r'$A@\Phi_{S}(f=%s Hz)$' % f0])
plt.xlabel("Position")
plt.xticks(ticks=np.arange(6), labels=[r'$0\degree$', r'$60\degree$', r'$120\degree$',r'$180\degree$',r'$240\degree$',r'$300\degree$'])
plt.show()
print("Hello World")

