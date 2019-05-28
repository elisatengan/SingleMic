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
npos = 4
delta_theta = 2*math.pi/4
theta_matrix = np.array([np.arange(360)*math.pi/180,]*npos)
theta_shifts = np.array([np.arange(npos)*delta_theta,]*theta_matrix.shape[1]).transpose()
theta_response = theta_matrix - theta_shifts

dbnorm = lambda x: 20*np.log10(np.abs(x)/np.max(x))
alpha = np.arange(0, 360, 1)
x = np.deg2rad(alpha)
y = mic_resp['cardioid'](theta_response[0,:])
ydb = dbnorm(y)
print(np.where(np.isnan(ydb)))
# plot
fig0 = plt.figure()
ax = plt.subplot(111, polar=True)
# set zero north
ax.set_theta_zero_location('N')
ax.set_theta_direction('counterclockwise')
ax.set_ylim(-30, 0)


plt.plot(x, ydb)
fig1 = plt.figure()
ax = plt.subplot(111, polar=True)
# set zero north
ax.set_theta_zero_location('N')
ax.set_theta_direction('counterclockwise')
ax.set_ylim(-30, 0)
plt.plot(x, dbnorm(mic_resp['cardioid'](theta_response[1,:])))
fig2 = plt.figure()
ax = plt.subplot(111, polar=True)
# set zero north
ax.set_theta_zero_location('N')
ax.set_theta_direction('counterclockwise')
ax.set_ylim(-30, 0)
plt.plot(x, dbnorm(mic_resp['cardioid'](theta_response[2,:])))
fig3 = plt.figure()
ax = plt.subplot(111, polar=True)
# set zero north
ax.set_theta_zero_location('N')
ax.set_theta_direction('counterclockwise')
ax.set_ylim(-30, 0)
plt.plot(x, dbnorm(mic_resp['cardioid'](theta_response[3,:])))
plt.show()

print("Hope World")