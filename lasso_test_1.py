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

from sklearn.linear_model import Lasso


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


def estimate_psd(signal, fs, method='periodogram', window='hanning', nperseg=256, noverlap=None):
    freq = None
    signal_psd = None
    if noverlap is None:
        noverlap = round(nperseg/2)

    if method == 'periodogram':
        signal_FT = np.fft.fft(signal,len(signal))/len(signal)
        freq = np.fft.fftfreq(len(signal))*fs
        signal_psd = np.power(abs(signal_FT),2)/(fs/len(signal))

    if method == 'bartlett':
        nsegs = round(len(signal) / nperseg)
        seg_FT = np.empty((nperseg, nsegs), dtype=complex)
        for i in range(nsegs):
            segment = signal[i * nperseg:(i + 1) * nperseg]
            seg_FT[:, i] = np.fft.fft(segment, nperseg)/nperseg
        seg_psd = np.power(abs(seg_FT), 2)/(fs/nperseg)
        freq = np.fft.fftfreq(nperseg)*fs
        signal_psd = np.mean(seg_psd,axis=1)

    if method == 'welch':
        hann_window = scipy.signal.get_window(window, nperseg)

        S1 = (abs(sum(hann_window))) ** 2

        S2 = (hann_window * hann_window).sum()
        ENBW = fs * (S2 / S1)

        nwindows = int(round(len(signal) - noverlap) / (nperseg - noverlap))
        noffset = nperseg - noverlap
        seg_FT = np.empty((nperseg, nwindows), dtype=complex)
        sig = scipy.signal.detrend(signal, type='constant')
        for i in range(nwindows):
            segment = np.multiply(sig[i * noffset:(i * noffset) + nperseg], hann_window)
            seg_FT[:, i] = (np.fft.fft(segment, nperseg)) / nperseg
        seg_psd = (np.power(abs(seg_FT), 2)) / ENBW
        freq = np.fft.fftfreq(nperseg) * fs
        signal_psd = np.mean(seg_psd, axis=1)
        # freq, signal_psd = scipy.signal.welch(signal, fs=fs, nperseg=nperseg, window=window, noverlap=noverlap)

    return freq, signal_psd


"""
Defining main parameters
"""
# Script for simulating whole model with speech samples.
fs = 16000  # sampling frequency
Nsec = 4  # time window in seconds
N = round(Nsec*fs)  # number of samples
tstep = (Nsec/N)  # time step between each sample
time = np.arange(N) * tstep
L = 2  # number of sources
T = 1000  # number of trials
alpha = 10  # factor multiplying pi for calculating the velocity
v = alpha*math.pi  # velocity rad/s
v_str = '%spi' % alpha  # string to specify velocity in simulation file
theta_moving = np.empty((N, L))  # varying source angles
A0 = 1
A1 = 1
phi0 = np.random.uniform(-math.pi, math.pi, 1)
phi1 = np.random.uniform(-math.pi, math.pi, 1)
f0 = 200
omega0 = 2*math.pi*f0
omega1 = 2*math.pi*f0
SNR = 20  # Signal to noise ratio in dB
delta_theta = math.pi/6
theta_sources = np.array([0 + delta_theta, delta_theta + (math.pi)])

print("phi0 = %s     phi1 = %s" % (phi0, phi1))
n_pos = 4  # number of microphone positions to be considered
segdur_sec = Nsec/n_pos
nsamples_seg = round(segdur_sec*fs)
theta_singleshift = 2*math.pi/n_pos
positions = np.arange(n_pos) * (2*math.pi/n_pos)
positions_aux = np.array([positions, ]*L).transpose()
theta_rotation = np.ones((n_pos, L)) * theta_sources
theta_rotation = theta_rotation - positions_aux
sig0 = A0 * np.exp(1j*(omega0*time + phi0))
sig1 = A1 * np.exp(1j*(omega1*time + phi1))
theta_matrix = np.array([np.arange(start=0,stop=360,step=15)*math.pi/180,]*n_pos)
theta_shifts = np.array([np.arange(n_pos)*theta_singleshift,]*theta_matrix.shape[1]).transpose()
theta_response = theta_matrix - theta_shifts

for i in range(n_pos):
    sig0[i*nsamples_seg:(i+1)*nsamples_seg] = mic_resp['cardioid'](theta_rotation[i,0]) * sig0[i*nsamples_seg:(i+1)*nsamples_seg]
    sig1[i * nsamples_seg:(i + 1) * nsamples_seg] = mic_resp['cardioid'](theta_rotation[i,1]) * sig1[i * nsamples_seg:(i + 1) * nsamples_seg]

var_sig0 = A1*A1  # statistics.variance(np.real(sig0))
var_noise = var_sig0/(math.pow(10, (SNR/10)))
noise_sig = np.random.normal(scale=np.sqrt(var_noise/2), size=(len(sig0),)) + 1j * np.random.normal(scale=np.sqrt(var_noise/2), size=(len(sig0),))  # + (np.random.uniform(0, np.sqrt(var_noise/2), size=(len(sig0),)))*1j
y = sig0 + sig1 + noise_sig

y_psd_hat_periodogram = np.empty((nsamples_seg,n_pos))
y_psd_hat_bartlett = np.empty((round(nsamples_seg/8),n_pos))
y_psd_hat_welch = np.empty((round(nsamples_seg/8),n_pos))
for i in range(n_pos):
    seg = y[i*nsamples_seg:(i+1)*nsamples_seg]
    freq, y_psd_hat_bartlett[:,i] = estimate_psd(seg, fs, 'bartlett', nperseg=round(nsamples_seg/8))
    freq, y_psd_hat_periodogram[:, i] = estimate_psd(seg, fs, 'periodogram')

nfreqs = len(freq)
X_train = theta_response
# y_train = y_psd_hat_periodogram[200,:]
y_train = np.array([0.5,2.,0.5,0])
test = X_train.T @ X_train
test = test/np.max(test)
lasso = Lasso(alpha=0.1)
lasso.fit(X_train,y_train)
train_score=lasso.score(X_train,y_train)
coeff_used = np.sum(lasso.coef_!=0)
print ("training score:", train_score)

print ("number of features used: ", coeff_used)
plt.plot(lasso.coef_,alpha=0.7,linestyle='none',marker='*',markersize=5,color='red',label=r'Lasso; $\alpha = 1$',zorder=7)
plt.show()