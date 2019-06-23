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

def _omni(x):
    return x

mic_resp = {
    'cardioid': _cardioid,
    'subcardioid': _subcardioid,
    'hypercardioid': _hypercardioid,
    'fig8': _fig8,
    'omni': _omni
}

"""
Defining main parameters
"""
# Script for simulating whole model with speech samples.
fs = 16000  # sampling frequency
Nsec = 0.02  # time window in seconds
N = round(Nsec*fs)  # number of samples
tstep = (Nsec/N)  # time step between each sample
time = np.arange(1024) * tstep
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
f0 = 1234
omega0 = 2*math.pi*f0
omega1 = 2*math.pi*f0
SNR = 0  # Signal to noise ratio in dB
theta_sources = np.array([0, math.pi])
print("phi0 = %s     phi1 = %s" % (phi0, phi1))

sig0 = A0 * np.exp(1j*(omega0*time + phi0))
sig1 = A1 * np.exp(1j*(omega1*time + phi1))

# sig0 = A0 * (np.sin(omega0*time + phi0))
# sig1 = A1 * (np.sin(omega1*time + phi1))


print(len(time))
var_sig0 = A1*A1  # statistics.variance(np.real(sig0))
var_noise = var_sig0/(math.pow(10, (SNR/10)))
noise_sig = np.random.normal(scale=np.sqrt(var_noise/2), size=(len(sig0),)) + 1j * np.random.normal(scale=np.sqrt(var_noise/2), size=(len(sig0),))  # + (np.random.uniform(0, np.sqrt(var_noise/2), size=(len(sig0),)))*1j
y = sig0 + noise_sig
"""
Elisa's Periodogram
"""
y_dft = np.fft.fft(y, n=1024)
fig0 = plt.figure()
freq = fs*np.fft.fftfreq(1024)
# plt.plot(freq, np.power(abs(y_dft),2)/(1024*fs))
plt.plot(freq, (np.power(abs(y_dft)/1024,2)))
#
#
fig1 = plt.figure()
f, Pxx_den = scipy.signal.periodogram(y, fs, nfft=1024,return_onesided=False)
plt.plot(f, Pxx_den)
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')


"""
Elisa's Bartlett method for psd estimation
"""
n = 512
segment = np.empty((n,))
seg_dft = np.empty((n, round(len(y)/n)),dtype=complex)
for i in range(round(len(y)/n)):
    segment = y[i*n:(i+1)*n]
    seg_dft[:,i] = np.fft.fft(segment, n)
seg_psd = (np.power((abs(seg_dft)/n),2))/(fs/n)
est_psd = np.mean(seg_psd, axis=1)


fig2 = plt.figure()
freq = fs*np.fft.fftfreq(n)
# plt.plot(freq, (est_psd*n)/(fs))
plt.plot(freq, est_psd)


fig3 = plt.figure()
f, Pxx_den = scipy.signal.welch(y, fs=fs, nperseg=n, window='boxcar',noverlap=0, return_onesided=False)
plt.plot(f, Pxx_den)
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.show()














