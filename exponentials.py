"""

Elisa Tengan Pires de Souza
KU Leuven
Department of Electrical Engineering (ESAT)

E-mail: elisa.tengan@esat.kuleuven.be

Stadius Center for Dynamical Systems, Signal Processing and Data Analytics (STADIUS)
Kasteelpark Arenberg 10
3001 Leuven (Heverlee)
Belgium



April 2019


"""

import numpy as np
import matplotlib.pyplot as plt
import math
import soundfile as sf
import scipy.signal
import statistics

def _cardioid(x):
    return 0.5 * (1 + np.cos(x))


def _subcardioid(x):
    return 0.25 * (3 + np.cos(x))


def _hypercardioid(x):
    return 0.25 * (1 + 3*np.cos(x))


def _fig8(x):
    return np.cos(x)


mic_resp = {
    'cardioid': _cardioid,
    'subcardioid': _subcardioid,
    'hypercardioid': _hypercardioid,
    'fig8': _fig8
}

"""
Defining main parameters
"""
# Script for simulating whole model with speech samples.
fs = 16000  # sampling frequency
Nsec = 0.02  # time window in seconds
N = round(Nsec*fs)  # number of samples
tstep = (Nsec/N)  # time step between each sample
time = np.arange(N*100) * tstep
L = 2  # number of sources
T = 1000  # number of trials
alpha = 10  # factor multiplying pi for calculating the velocity
v = alpha*math.pi  # velocity rad/s
v_str = '%spi' % alpha  # string to specify velocity in simulation file
theta_moving = np.empty((N, L))  # varying source angles
A0 = 1
A1 = 3
phi0 = np.random.uniform(-math.pi, math.pi, 1)
phi1 = np.random.uniform(-math.pi, math.pi, 1)
f0 = 50
omega0 = 2*math.pi*f0
omega1 = 2*math.pi*f0
SNR = 0  # Signal to noise ratio in dB
theta_sources = np.array([0, math.pi])
print("phi0 = %s     phi1 = %s" % (phi0, phi1))


"""
Setting up signals and STFT
"""
sig0 = A0 * np.exp(1j*(omega0*time + phi0))
sig1 = A1 * np.exp(1j*(omega1*time + phi1))
freq, t, sig0_STFT = scipy.signal.stft(sig0, fs, nperseg=N, return_onesided=False)
freq, t, sig1_STFT = scipy.signal.stft(sig1, fs, nperseg=N, return_onesided=False)
S = np.array([sig0_STFT, sig1_STFT])

# Generate noise
var_sig0 = A1*A1  # statistics.variance(np.real(sig0))
var_noise = var_sig0/(math.pow(10, (SNR/10)))
noise_sig = np.random.normal(scale=np.sqrt(var_noise/2), size=(len(sig0),)) + 1j * np.random.normal(scale=np.sqrt(var_noise/2), size=(len(sig0),))  # + (np.random.uniform(0, np.sqrt(var_noise/2), size=(len(sig0),)))*1j
_, _, noise_STFT = scipy.signal.stft(noise_sig, fs, nperseg=N,return_onesided=False)

"""
Compute microphone output and then estimate source signals
"""

# Choose microphone response pattern
pattern = 'cardioid'
M = np.empty((len(t), L))  # microphone response matrix

# Allocate arrays for microphone output and estimated source signals
Y = np.zeros((len(t), len(freq)), dtype=complex)
S_hat = np.zeros((L, len(freq)), dtype=complex)

# Compute output and estimate S
for k in range(len(t)):  # for each STFT window k:
    delta_t = t[k]  # initial time
    for j in range(L): # for each source j:
        theta_moving[k, j] = theta_sources[j] + (v * (delta_t))
        # Matrix M's elements corresponds to microphone response values for each source for each observation window
        M[k, j] = mic_resp[pattern](theta_moving[k, j])
for w in range(len(freq)):  # for each frequency w:
    phase_shift = (np.array([np.exp(-1j * 2 * math.pi * freq[w] * t), ] * L)).transpose()
    M_new = M*phase_shift
    for k in range(len(t)):
        Y[k, w] = M[k, :] @ S[:, w, k] + noise_STFT[w, k]
    S_hat[:, w] = np.linalg.lstsq(M_new, Y[:, w], rcond=None)[0]  # source signals STFT calculated via LS


S_hat_sig0 = S_hat[0, :]
S_hat_sig1 = S_hat[1, :]

"""
Reordering arrays such that frequency is in increasing order
"""
ordered_freq = np.roll(freq,round(len(freq)/2))
ordered_S_hat_sig0 = np.roll(S_hat_sig0, round(len(freq)/2))
ordered_S_hat_sig1 = np.roll(S_hat_sig1, round(len(freq)/2))
ordered_noise_STFT = np.roll(noise_STFT, round(len(freq)/2))
ordered_Y = np.roll(Y,round(len(freq)/2),1)
ordered_sig0_STFT = np.roll(sig0_STFT, round(len(freq)/2))
ordered_sig1_STFT = np.roll(sig1_STFT, round(len(freq)/2))

"""
Plotting STFTs and attempted reconstruction of source signals
"""
fig1 = plt.figure()
plt.plot(ordered_freq, np.abs(ordered_S_hat_sig0))
plt.plot(ordered_freq, np.abs(ordered_S_hat_sig1))
plt.show()

# fig2 = plt.figure()
# plt.plot(ordered_freq, np.angle(ordered_S_hat_sig0))
# plt.plot(ordered_freq, np.angle(ordered_S_hat_sig1))
# plt.show()

idx_f0 = np.where(freq==f0)[0][0]
print("S0_hat = %s     S1_hat = %s  " % (S_hat_sig0[idx_f0], S_hat_sig1[idx_f0]))
print("phi0_hat = %s     phi1_hat = %s" % (np.angle(S_hat_sig0[idx_f0]), np.angle(S_hat_sig1[idx_f0])))



