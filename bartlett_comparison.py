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

"""
Signal generation
"""
# Setting parameters
fs = 16000  # sampling frequency
Nsec = 0.02  # time window in seconds
N = round(Nsec*fs)  # number of samples
tstep = (Nsec/N)  # time step between each sample
time = np.arange(1024) * tstep
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

var_sig0 = A1*A1  # statistics.variance(np.real(sig0))
var_noise = var_sig0/(math.pow(10, (SNR/10)))
noise_sig = np.random.normal(scale=np.sqrt(var_noise/2), size=(len(sig0),)) + 1j * np.random.normal(scale=np.sqrt(var_noise/2), size=(len(sig0),))  # + (np.random.uniform(0, np.sqrt(var_noise/2), size=(len(sig0),)))*1j
y = sig0 + noise_sig

"""
Computing and plotting PSD estimation. Bartlett method is equivalent to python's Welch implementation using 'boxcar' as
a window and no overlap
"""

n = 32
f, Pxx_den_32 = scipy.signal.welch(y, fs=fs, nperseg=n, window='boxcar', noverlap=0, return_onesided=False)
f = np.fft.fftshift(f)
Pxx_den_32 = np.fft.fftshift(Pxx_den_32)
plt.plot(f, Pxx_den_32, label='n = 32')
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD')

n = 64
f, Pxx_den_64 = scipy.signal.welch(y, fs=fs, nperseg=n, window='boxcar', noverlap=0, return_onesided=False)
f = np.fft.fftshift(f)
Pxx_den_64 = np.fft.fftshift(Pxx_den_64)
plt.plot(f, Pxx_den_64, label='n = 64')
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD')


n = 128
f, Pxx_den_128 = scipy.signal.welch(y, fs=fs, nperseg=n, window='boxcar', noverlap=0, return_onesided=False)
f = np.fft.fftshift(f)
Pxx_den_128 = np.fft.fftshift(Pxx_den_128)
plt.plot(f, Pxx_den_128, label='n = 128')
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD')

n = 256
f, Pxx_den_256 = scipy.signal.welch(y, fs=fs, nperseg=n, window='boxcar', noverlap=0, return_onesided=False)
f = np.fft.fftshift(f)
Pxx_den_256 = np.fft.fftshift(Pxx_den_256)
plt.plot(f, Pxx_den_256, label='n = 256')
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD')
plt.legend()
plt.show()
