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
Nsec = 0.02 # time window in seconds
N = round(Nsec*fs)  # number of samples
tstep = (Nsec/N)  # time step between each sample
time = np.arange(N*100) * tstep
L = 1  # number of sources
T = 1000  # number of trials
alpha = 10  # factor multiplying pi for calculating the velocity
v = alpha*math.pi  # velocity rad/s
v_str = '%spi' % alpha  # string to specify velocity in simulation file
theta_moving = np.empty((N, L))  # varying source angles
A0 = 1
# phi0 = np.random.uniform(0, 2*math.pi, 1)[0]
phi0 = math.pi/2
f00 = 50
f0 = 50  # Phase shift is visible for low frequencies
omega0 = 2*math.pi*f00
SNR = 0  # Signal to noise ratio in dB
theta_sources = 0  # one source at 0 degrees
"""
Setting up signals and STFT
"""
sig0 = A0 * np.exp(1j*(omega0*time + phi0))
freq, t, sig0_STFT = scipy.signal.stft(sig0, fs, nperseg=N, return_onesided=False)
idx_f0 = np.where(freq == f0)[0][0]  # index of signal's frequency in STFT freq array
S = sig0_STFT


# Generate noise
var_noise = (A0*A0)/(math.pow(10, (SNR/10)))
noise_sig = np.random.normal(scale=np.sqrt(var_noise/2), size=(len(sig0),)) + 1j * np.random.normal(scale=np.sqrt(var_noise/2), size=(len(sig0),))  # + (np.random.uniform(0, np.sqrt(var_noise/2), size=(len(sig0),)))*1j
_, _, noise_STFT = scipy.signal.stft(noise_sig, fs, nperseg=N, return_onesided=False)


"""
Compute microphone output
"""

# Choose microphone response pattern
pattern = 'cardioid'
M = np.empty((len(t),), dtype=complex)  # microphone response matrix
theta_moving = np.empty((len(t), ))

# Allocate arrays for microphone output and estimated source signals
Y = np.zeros((len(t), len(freq)), dtype=complex)
S_hat = np.zeros((L, len(freq)), dtype=complex)

# Compute output
for k in range(len(t)):  # for each STFT window k:
    delta_t = t[k]  # initial time
    theta_moving[k] = theta_sources + (v * (delta_t))
    # Matrix M's elements corresponds to microphone response values for each source for each observation window
    M[k] = mic_resp[pattern](theta_moving[k])
for w in range(len(freq)):  # for each frequency w:
    phase_shift = (np.array([np.exp(-1j * 2 * math.pi * freq[w] * t), ] * L)).transpose()
    M_new = M*phase_shift
    for k in range(len(t)):
        Y[k, w] = M[k] * S[w, k]  # + noise_STFT[w, k]
    # S_hat[:, w] = np.linalg.lstsq(M, Y[:, w], rcond=None)[0]


# S_hat_sig0 = S_hat[0, :]

"""
Reordering arrays such that frequency is in increasing order
"""

ordered_freq = np.roll(freq,round(len(freq)/2))
ordered_noise_STFT = np.roll(noise_STFT, round(len(freq)/2))
ordered_Y = np.roll(Y, round(len(freq)/2), 1)
ordered_sig0_STFT = np.roll(sig0_STFT, round(len(freq)/2))

"""
Plotting magnitude and phase of source signal and microphone output
"""

fig = plt.figure()
plt.plot(t, (np.abs(S[idx_f0])).transpose())
plt.xlabel('Time (s)')
plt.ylabel(r'$|S(\omega = \omega_0)|$')

x = (np.angle(S[idx_f0]))

fig0 = plt.figure()
plt.plot(t, x)
plt.xlabel('Time (s)')
plt.ylabel(r'$\angle S(\omega = \omega_0)$ (rad)')

fig2 = plt.figure()
plt.plot(t, (np.abs(Y[:,idx_f0])))
plt.xlabel('Time (s)')
plt.ylabel(r'$|Y(\omega = \omega_0)|$')

fig3 = plt.figure()
plt.plot(t, (np.angle(Y[:,idx_f0])))
plt.xlabel('Time (s)')
plt.ylabel(r'$\angle Y(\omega = \omega_0)$ (rad)')



plt.show()



