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

n = 256
noverlap = 128
hann_window = scipy.signal.get_window("hanning", n)

S1 = (abs(sum(hann_window)))**2

S2 = (hann_window*hann_window).sum()
ENBW = fs*(S2/S1)

nwindows = int(round(len(y) - noverlap)/(n - noverlap))
noffset = n - noverlap
seg_FT = np.empty((n, nwindows), dtype=complex)
seg_psd = np.empty((n, nwindows))
sig = scipy.signal.detrend(y,type='constant')
for i in range(nwindows):
    segment = np.multiply(sig[i*noffset:(i*noffset) + n], hann_window)
    seg_FT[:,i] = (np.fft.fft(segment,n))/n
seg_psd = (np.power(abs(seg_FT), 2))/ENBW
freq = np.fft.fftfreq(n)*fs
signal_psd = np.mean(seg_psd, axis=1)


f, Pxx_den_256 = scipy.signal.welch(y, fs=fs, nperseg=n, window='hanning', return_onesided=False)

plt.plot(freq, signal_psd)
plt.plot(f, Pxx_den_256*S1/(n**2))

# f, Pxx_den_256 = scipy.signal.welch(y, fs=fs, nperseg=n, window='flattop', return_onesided=False)
# plt.plot(f, Pxx_den_256)
#
# f, Pxx_den_256 = scipy.signal.welch(y, fs=fs, nperseg=n, window='hamming', return_onesided=False)
# plt.plot(f, Pxx_den_256)
#
# f, Pxx_den_256 = scipy.signal.welch(y, fs=fs, nperseg=n, window='blackman', return_onesided=False)
# plt.plot(f, Pxx_den_256)


plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.legend(["Elisa", "Scipy"])

plt.show()
