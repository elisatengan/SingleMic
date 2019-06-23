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
import numpy as np
import matplotlib.pyplot as plt
from signal_generator import *
from psd_estimator import *
from noise_generator import *
from sklearn.linear_model import Lasso
from grouplasso import GroupLassoRegressor
from pyglmnet import *
import scipy.io as io
print("Hello lasso_test_noise.py")

"""
Signal generation
"""

# Setting parameters

fs = 16000
nb_sources = 1
duration_sig = 6*((2**14)/16000)  # I am cheating in order to get frames with size as a power of 2
Nsamples_total = int(round((duration_sig*fs)))
n_pos = 6
thetas = np.array([30*math.pi/180])
pwr = np.array([1])
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

# fig = plt.figure()
# plt.plot(np.sum(y_psd_hat_periodogram, axis=0))
# plt.show()
"""
Cheating output
"""
theta_m=0
delta_theta = 2*math.pi/n_pos

# Auxiliary matrices with angles for creating coefficient matrix
mystep = 1
theta_matrix = np.array([np.arange(360, step=mystep)*math.pi/180,]*n_pos)
theta_shifts = np.array([np.arange(n_pos)*delta_theta,]*theta_matrix.shape[1]).transpose()
theta_response = theta_matrix - theta_shifts - theta_m

# Cheating PSD estimation of sources
welch_nperseg = 512
welch_overlap = 256
psd_S = np.empty((welch_nperseg,nb_sources))
for i in range(nb_sources):
    freq,psd_S[:,i] = estimate_psd(sources_signals[i,0:nsamples_seg],fs,method='welch', nperseg=welch_nperseg, noverlap=welch_overlap)

# Creating vector with sources' PSD for a single frequency. We know that sources are at 0 and 180 degrees

# x = np.zeros((len(freq)*int(360/mystep),))
# x[0:len(freq)] = psd_S[:,0]
# x[len(freq)*int(180/mystep):len(freq)*(int(180/mystep)+1)] = psd_S[:,1]

idx_freq = 200
# x_single = np.zeros((int(360/mystep),))
# x_single[0] = psd_S[idx_freq, 0]
# x_single[int(180/mystep)] = psd_S[idx_freq, 1]

# Finally, cheating PSD estimation of microphone output
matrix_A = np.power(mic_resp['cardioid'](theta_response), 2)

fig0 = plt.figure()
plt.imshow(matrix_A)
plt.show()

big_matrix_A = np.repeat(matrix_A, len(freq), axis=1)

# test_y = big_matrix_A @ x
# test_single_y = matrix_A @ x_single
# fig = plt.figure()
# plt.plot(y_psd_hat_periodogram[idx_freq, :],label='output')
# plt.plot(test_single_y,label='cheat')
# plt.legend()
#
# fig1 = plt.figure()
# plt.plot(np.sum(y_psd_hat_periodogram, axis=0),label='output')
# plt.plot(test_y,label='cheat')
# plt.legend()


# y_psd_cheat_singlefreq = matrix_A @ x
# y_psd_periodogram_singlefreq = y_psd_hat_periodogram[idx_freq, :]
# y_train = np.sum(y_psd_hat_periodogram, axis=0)
# fig = plt.figure()
# plt.plot(test_y, label=r'$A\times s$')
# plt.plot(y_train, label=r'Y')
# plt.xticks(ticks=[0,6,11], labels=[r'$0\degree$', r'$180\degree$', r'$330\degree$'])
# plt.xlim(0,11)
# plt.legend()
#
# plt.xlabel('Position')

# y_train= y_psd_hat_periodogram.reshape((y_psd_hat_periodogram.size,), order='F')
y_train = y_psd_hat[idx_freq,:]
# np.sum(y_psd_hat_periodogram,axis=0)
id_array = np.arange(360, step=mystep)
group_ids = np.repeat(id_array, len(freq))
X_train = matrix_A
# io.savemat("data_grouplasso_test_yall1_3sources.mat",dict([("A", matrix_A), ("b", y_train), ("ids", group_ids)]))
# model = GroupLassoRegressor(group_ids=group_ids, random_state=42, verbose=False, alpha=0.00001)
# model.fit(X_train, y_train)
print("finish?")

lasso = Lasso(alpha=0.00001)
lasso.fit(X_train,y_train)
# train_score=lasso.score(X_train,y_train)
# coeff_used = np.sum(lasso.coef_!=0)
# print ("training score:", train_score)
#
# print ("number of features used: ", coeff_used)
plt.figure()
plt.plot(lasso.coef_,alpha=0.7,linestyle='none',marker='*',markersize=5,color='red',label=r'Lasso; $\alpha = 1$',zorder=7)
plt.show()
