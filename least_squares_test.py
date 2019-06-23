"""

Elisa Tengan Pires de Souza
KU Leuven
Department of Electrical Engineering (ESAT)

E-mail: elisa.tengan@esat.kuleuven.be

Stadius Center for Dynamical Systems, Signal Processing and Data Analytics (STADIUS)
Kasteelpark Arenberg 10
3001 Leuven (Heverlee)
Belgium



June 2019


"""

import matplotlib.pyplot as plt
from signal_generator import *
from psd_estimator import *
from noise_generator import *

"""
Signal generation - 1 source
"""

# Setting parameters

fs = 16000
nb_sources = 1
duration_sig = 6*((2**14)/16000)  # I am cheating in order to get frames with size as a power of 2
Nsamples_total = int(round((duration_sig*fs)))
n_pos = 12

# Generating signals
angles_sources, sources_signals = generate_noise(nb_sources, fs, duration_sig, positions=np.array([math.pi]), noise_power=np.array([0.3]))
nsamples_seg, delta_theta, y = generate_mic_output(sources_signals, angles_sources, n_pos, 'cardioid', theta_m=0)

"""
Estimating the PSD of the microphone's output signal
"""
welch_nperseg = 512
welch_overlap = 256
y_psd_hat = np.empty((welch_nperseg, n_pos))
for i in range(n_pos):
    seg = y[i*nsamples_seg:(i+1)*nsamples_seg]
    freq, y_psd_hat[:, i] = estimate_psd(seg, fs, method='welch', nperseg=welch_nperseg, noverlap=welch_overlap)


"""
Coefficient matrix
"""

theta_m=0
delta_theta = 2*math.pi/n_pos

# Auxiliary matrices with angles for creating coefficient matrix
mystep = 3  # angle step in degrees -> L = 120
L = int(360/mystep)
theta_matrix = np.array([np.arange(360, step=mystep)*math.pi/180,]*n_pos)
theta_shifts = np.array([np.arange(n_pos)*delta_theta,]*theta_matrix.shape[1]).transpose()
theta_response = theta_matrix - theta_shifts - theta_m

matrix_A = np.power(mic_resp['cardioid'](theta_response), 2)

"""
Solving system for a single frequency
"""
freq_idx = 256  # select some frequency
A = np.asmatrix(matrix_A)
b = y_psd_hat[freq_idx,:]

x = np.linalg.solve((matrix_A.T)@matrix_A + 1*np.identity(L),(matrix_A.T)@b)

x1 = np.linalg.lstsq(matrix_A,b,rcond=None)

fig = plt.figure(figsize=(16,6))
plt.plot(x, label='Elisa')
plt.plot(x1[0],label='linalg')
plt.xlabel("Position", fontsize=14)
plt.ylabel("LS coefficient", fontsize=14)
plt.legend()
plt.xticks(ticks=[0,30,60,90], labels=[r'$0\degree$', r'$90\degree$', r'$180\degree$', r'$270\degree$'])
plt.xlim(0,len(x)-1)
plt.show()
