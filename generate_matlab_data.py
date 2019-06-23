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

from signal_generator import *
from psd_estimator import *
from noise_generator import *
import scipy.io as io
print("Hello generate_matlab_data.py")

"""
Signal generation
"""

# Setting parameters

fs = 16000
nb_sources = 5
duration_sig = 6*((2**14)/16000)  # I am cheating in order to get frames with size as a power of 2
Nsamples_total = int(round((duration_sig*fs)))
n_pos = 6
thetas = 'uniform'
pwr = np.array([1, 1, 1, 1, 1])
# Generating signals
angles_sources, sources_signals = generate_noise(nb_sources, fs, duration_sig, thetas, pwr)
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
Creating matrix A
"""
theta_m=0
delta_theta = 2*math.pi/n_pos

# Auxiliary matrices with angles for creating coefficient matrix
mystep = 1
theta_matrix = np.array([np.arange(360, step=mystep)*math.pi/180,]*n_pos)
theta_shifts = np.array([np.arange(n_pos)*delta_theta,]*theta_matrix.shape[1]).transpose()
theta_response = theta_matrix - theta_shifts - theta_m

# Finally, cheating PSD estimation of microphone output
matrix_A = np.power(mic_resp['cardioid'](theta_response), 2)
idx_freq = 200

idx_freq = 200

y_train = y_psd_hat[idx_freq,:]

io.savemat("200619_data_lasso_singlefreq_L=%s.mat" % nb_sources, dict([("A", matrix_A), ("b", y_train)]))

