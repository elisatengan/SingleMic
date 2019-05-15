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
A1 = 3
phi0 = np.random.uniform(-math.pi, math.pi, 1)
phi1 = np.random.uniform(-math.pi, math.pi, 1)
f0 = 10
omega0 = 2*math.pi*f0
omega1 = 2*math.pi*f0
SNR = 50  # Signal to noise ratio in dB
delta_theta = math.pi/6
theta_sources = np.array([0 + delta_theta, delta_theta + (math.pi)])

print("phi0 = %s     phi1 = %s" % (phi0, phi1))
n_pos = 4  # number of microphone positions to be considered
segdur_sec = Nsec/n_pos
nsamples_seg = round(segdur_sec*fs)
positions = np.arange(n_pos) * (2*math.pi/n_pos)
positions_aux = np.array([positions, ]*L).transpose()
theta_rotation = np.ones((n_pos, L)) * theta_sources
theta_rotation = theta_rotation - positions_aux
sig0 = A0 * np.exp(1j*(omega0*time + phi0))
sig1 = A1 * np.exp(1j*(omega1*time + phi1))


for i in range(n_pos):
    sig0[i*nsamples_seg:(i+1)*nsamples_seg] = mic_resp['cardioid'](theta_rotation[i,0]) * sig0[i*nsamples_seg:(i+1)*nsamples_seg]
    sig1[i * nsamples_seg:(i + 1) * nsamples_seg] = mic_resp['cardioid'](theta_rotation[i,1]) * sig1[i * nsamples_seg:(i + 1) * nsamples_seg]

var_sig0 = A1*A1  # statistics.variance(np.real(sig0))
var_noise = var_sig0/(math.pow(10, (SNR/10)))
noise_sig = np.random.normal(scale=np.sqrt(var_noise/2), size=(len(sig0),)) + 1j * np.random.normal(scale=np.sqrt(var_noise/2), size=(len(sig0),))  # + (np.random.uniform(0, np.sqrt(var_noise/2), size=(len(sig0),)))*1j
y = sig0 + sig1 + noise_sig

