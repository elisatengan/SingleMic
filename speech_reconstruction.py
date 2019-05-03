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
import matplotlib
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
N = round(Nsec*fs)  # number of samples for one STFT window
tstep = (Nsec/N)  # time step between each sample
time = np.arange(N) * tstep  # time samples array
L = 3  # number of sources
T = 1000  # number of trials
alpha = 1  # factor multiplying pi for calculating the angular velocity
v = alpha*math.pi  # angular velocity rad/s
v_str = '%spi' % alpha  # string to specify velocity in simulation file
M = np.empty((N, L))  # microphone response matrix
theta_moving = np.empty((N, L))  # source angles varying with time due to microphone rotation
SNR = 10  # Signal to noise ratio in dB (related to sig0's power)
# Create array with source angles
theta_sources = np.linspace(0, 2 * math.pi * (1 - 1 / L), L)  # sources equally distanced from each other


"""
Setting up signals and STFT
"""
# Read speech sample files
sig0, fsig0 = sf.read('FA01_01.wav')
sig1, fsig1 = sf.read('MB07_01.wav')
sig2, fsig2 = sf.read('CA01_03.wav')  # Third signal in case it is chosen to have 3 sources


# Match length of speech signals and then calculate STFT
if L == 2:

    length_diff = len(sig0) - len(sig1)
    if length_diff > 0:
        sig1 = np.concatenate((sig1, np.zeros(length_diff)))
    elif length_diff < 0:
        sig0 = np.concatenate((sig0, np.zeros(length_diff)))

    freq, t, sig0_STFT = scipy.signal.stft(sig0, fsig0, nperseg=N)
    freq, t, sig1_STFT = scipy.signal.stft(sig1, fsig1, nperseg=N)

    S_orig = np.array([sig0_STFT, sig1_STFT])

if L == 3:

    max_len = max(np.array([len(sig0), len(sig1), len(sig2)]))

    length_diff_0 = max_len - len(sig0)

    if length_diff_0 > 0:
        sig0 = np.concatenate((sig0, np.zeros(length_diff_0)))

    length_diff_1 = max_len - len(sig1)

    if length_diff_1 > 0:
        sig1 = np.concatenate((sig1, np.zeros(length_diff_1)))

    length_diff_2 = max_len - len(sig2)

    if length_diff_2 > 0:
        sig2 = np.concatenate((sig2, np.zeros(length_diff_2)))

    freq, t, sig0_STFT = scipy.signal.stft(sig0, fsig0, nperseg=N)
    freq, t, sig1_STFT = scipy.signal.stft(sig1, fsig1, nperseg=N)
    freq, t, sig2_STFT = scipy.signal.stft(sig2, fsig2, nperseg=N)
    S_orig = np.array([sig0_STFT, sig1_STFT, sig2_STFT])

# Generate noise
var_sig0 = statistics.variance(sig0)
var_noise = var_sig0/(math.pow(10, (SNR/10)))
noise_sig = np.random.normal(scale=np.sqrt(var_noise), size=(len(sig0),))
_, _, noise_STFT = scipy.signal.stft(noise_sig, fsig0, nperseg=N)


"""
Compute microphone output and then estimate source signals
"""

# Choose microphone response pattern
pattern = 'cardioid'

# Allocate arrays for microphone output and estimated source signals
Y = np.zeros((N, len(freq), len(t)), dtype=complex)  # reused for each each STFT window
S_hat = np.zeros((L, len(freq), len(t)), dtype=complex)

# Compute output and estimate S
for k in range(len(t)):  # for each STFT window k:
    t0 = t[k]  # initial time
    for i in range(N):  # for each time instant i:
        for j in range(L):  # for each source j:
            theta_moving[i, j] = theta_sources[j] + (v * (t0 + (i * tstep)))
            # Matrix M's elements corresponds to microphone response values for each source at each instant
            M[i, j] = mic_resp[pattern](theta_moving[i, j])
    # print("condition: %s      rank: %s" % (np.linalg.cond(M), np.linalg.matrix_rank(M)))
    for w in range(len(freq)):  # for each frequency w:
        # I think this calculation is wrong because I am adding the same noise value for each element of vector Y
        Y[:, w, k] = M @ S_orig[:, w, k] + noise_STFT[w, k] * np.ones(N)

        S_hat[:, w, k] = np.linalg.lstsq(M, Y[:, w, k], rcond=None)[0]  # source signals STFT calculated via LS

S_hat_sig0 = S_hat[0, :, :]
S_hat_sig1 = S_hat[1, :, :]


"""
Reconstruct time signals
"""
_, sig0_hat = scipy.signal.istft(S_hat_sig0, fsig0)
_, sig1_hat = scipy.signal.istft(S_hat_sig1, fsig1)
# sf.write('sig0_hat_v1_L=%s_v=%s_mic=%s.wav' % (L, v_str, pattern), sig0_hat,16000)
# sf.write('sig1_hat_v1_L=%s_v=%s_mic=%s.wav' % (L, v_str, pattern), sig1_hat, 16000)

if L == 3:
    S_hat_sig2 = S_hat[2, :, :]
    _, sig2_hat = scipy.signal.istft(S_hat_sig2, fsig2)
    # sf.write('sig2_hat_v1_L=%s_v=%s_mic=%s.wav' % (L, v_str, pattern), sig2_hat, 16000)


"""
Plotting STFTs and attempted reconstruction of source signals
"""


fig000, (ax1, ax2) = plt.subplots(1, 2)
im = ax1.pcolormesh(t, freq, np.abs(sig0_STFT), vmin=0)
im = ax2.pcolormesh(t, freq, np.abs(S_hat_sig0), vmin=0)
ax1.set_title(r'$|S_0|$')
ax2.set_title(r'$|\hat{S}_0|$')
ax1.set_ylabel('Frequency (Hz)')
ax2.set_ylabel('Frequency (Hz)')
ax1.set_xlabel('Time (sec)')
ax2.set_xlabel('Time (sec)')
plt.colorbar(im, ax=(ax1, ax2))
fig000.suptitle('STFT Magnitude')


fig111, (ax1, ax2) = plt.subplots(1, 2)
im = ax1.pcolormesh(t, freq, np.abs(sig1_STFT), vmin=0)
im = ax2.pcolormesh(t, freq, np.abs(S_hat_sig1), vmin=0)
ax1.set_title(r'$|S_1|$')
ax2.set_title(r'$|\hat{S}_1|$')
ax1.set_ylabel('Frequency (Hz)')
ax2.set_ylabel('Frequency (Hz)')
ax1.set_xlabel('Time (sec)')
ax2.set_xlabel('Time (sec)')
plt.colorbar(im, ax=(ax1, ax2))
fig111.suptitle('STFT Magnitude')


if L == 3:
    S_hat_sig2 = S_hat[2, :, :]
    fig222, (ax1, ax2) = plt.subplots(1, 2)
    im = ax1.pcolormesh(t, freq, np.abs(sig2_STFT), vmin=0)
    im = ax2.pcolormesh(t, freq, np.abs(S_hat_sig2), vmin=0)
    ax1.set_title(r'$|S_2|$')
    ax2.set_title(r'$|\hat{S}_2|$')
    ax1.set_ylabel('Frequency (Hz)')
    ax2.set_ylabel('Frequency (Hz)')
    ax1.set_xlabel('Time (sec)')
    ax2.set_xlabel('Time (sec)')
    plt.colorbar(im, ax=(ax1, ax2))
    fig222.suptitle('STFT Magnitude')


plt.show()




