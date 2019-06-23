import numpy as np
import matplotlib.pyplot as plt
from psd_estimator import *

# This is an adaptation of Maja's test script.

"""
Signal generation
"""

# Setting parameters

fs = 16000
winlen = 512
overlap = int(winlen/2)
p1 = 0.4
p2 = 0.3

duration = 30  # seconds
numsamples = duration*fs

s1 = np.sqrt(p1)*np.random.randn(numsamples,)
s2 = np.sqrt(p2)*np.random.randn(numsamples,)

# The microphone signal (for now just a sum of the two, no direction-dependent gain)
y = s1 + s2

# Total number of overlapping windows in time
nwindows = int((len(y) - overlap)/(winlen - overlap))

psds1 = np.empty((winlen, nwindows))
psds2 = np.empty((winlen, nwindows))
psdsy = np.empty((winlen, nwindows))
tmp1 = np.zeros(shape=(winlen,))
tmp2 = np.zeros(shape=(winlen,))
tmpy = np.zeros(shape=(winlen,))

# PSD estimation for each segment
for idx in range(nwindows):
    freq_psd1, psds1[:, idx] = estimate_psd(s1[idx * (winlen - overlap):idx * (winlen-overlap) + winlen], fs, method='welch', nperseg=winlen, noverlap=0)
    freq_psd2, psds2[:, idx] = estimate_psd(s2[idx * (winlen - overlap):idx * (winlen-overlap) + winlen], fs, method='welch', nperseg=winlen, noverlap=0)
    freq_psdy, psdsy[:, idx] = estimate_psd(y[idx * (winlen - overlap):idx * (winlen-overlap) + winlen], fs, method='welch', nperseg=winlen, noverlap=0)


L = 64  # number of segments to consider for averaging over

numfreq = len(freq_psdy)
psds1_MA = np.empty((numfreq, nwindows))
psds2_MA = np.empty((numfreq, nwindows))
psdsy_MA = np.empty((numfreq, nwindows))

psds1_exp = np.empty((numfreq, nwindows))
psds2_exp = np.empty((numfreq, nwindows))
psdsy_exp = np.empty((numfreq, nwindows))

tmp1 = np.zeros(shape=(numfreq,))
tmp2 = np.zeros(shape=(numfreq,))
tmpy = np.zeros(shape=(numfreq,))


taxis = np.arange(numsamples, step=(winlen-overlap))/fs
taxis = taxis[0:len(taxis)-1]

alpha = 0.95  # Exponential averaging factor

for idx in range(nwindows):
    if idx < L-1:
        psds1_MA[:, idx] = np.sum(psds1[:, 0:idx + 1], axis=1) / L
        psds2_MA[:, idx] = np.sum(psds2[:, 0:idx + 1], axis=1) / L
        psdsy_MA[:, idx] = np.sum(psdsy[:, 0:idx + 1], axis=1) / L

    else:
        psds1_MA[:, idx] = np.mean(psds1[:, idx - (L - 1):idx + 1], axis=1)
        psds2_MA[:, idx] = np.mean(psds2[:, idx - (L - 1):idx + 1], axis=1)
        psdsy_MA[:, idx] = np.mean(psdsy[:, idx - (L - 1):idx + 1], axis=1)

    # Exponential averaging
    tmp1 = alpha*tmp1 + (1-alpha)*(psds1[:, idx])
    tmp2 = alpha*tmp2 + (1-alpha)*(psds2[:, idx])
    tmpy = alpha*tmpy + (1-alpha)*(psdsy[:, idx])

    psds1_exp[:, idx] = tmp1
    psds2_exp[:, idx] = tmp2
    psdsy_exp[:, idx] = tmpy


# Select an arbitrary frequency to look at
selectfrequency = 200

fig1 = plt.figure(figsize = (15,6))
h1, = plt.plot(taxis, psds1_MA[selectfrequency, :], label=r'$\phi_{S1}$')
h2, = plt.plot(taxis, psds2_MA[selectfrequency, :], label=r'$\phi_{S2}$')
h3, = plt.plot(taxis, psdsy_MA[selectfrequency, :], label=r'$\phi_{Y}$')
h4, = plt.plot(taxis, psds1_MA[selectfrequency, :] + psds2_MA[selectfrequency, :], label=r'$\phi_{S1}$ + $\phi_{S2}$')
plt.legend()
plt.xlim(0,30)
plt.xlabel('time [seconds]', fontsize=14)
plt.ylabel('PSD estimate at one frequency', fontsize=14)

# Comparing Exp averaging with Welch
fig2 = plt.figure(figsize=(15, 6))
h1, = plt.plot(taxis, psdsy_exp[selectfrequency, :], label=r'$\phi_{Y}$ exp')
h2, = plt.plot(taxis, psdsy_MA[selectfrequency, :], label=r'$\phi_{Y}$ MA')
# h3, = plt.plot(taxis,psds2_exp[selectfrequency,:], label='PSD s2 exp')
# h4, = plt.plot(taxis,psds2_welch[selectfrequency,:], label='PSD s2 welch')
# h5, = plt.plot(taxis,psdsy_exp[selectfrequency,:], label='PSD y exp')
# h6, = plt.plot(taxis,psdsy_welch[selectfrequency,:], label='PSD y welch')
# h7, = plt.plot(taxis,psds1_exp[selectfrequency,:] + psds2_exp[selectfrequency,:], label='PSD s1 + PSD s2 exp')
# h8, = plt.plot(taxis,psds1_welch[selectfrequency,:] + psds2_welch[selectfrequency,:], label='PSD s1 + PSD s2 welch')
plt.legend()
plt.xlim(0,30)
plt.xlabel('time [seconds]', fontsize=14)
plt.ylabel('PSD estimate at one frequency', fontsize=14)

plt.show()
