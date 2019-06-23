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

n_positions = np.arange(start=4,stop=13)
mystep=np.arange(start=1,stop=31)
eig_matrix = np.empty((len(n_positions),len(mystep)),dtype=object)
eigv_matrix = np.empty((len(n_positions),len(mystep)),dtype=object)
theta_m=0

store_rank = np.empty((len(n_positions),len(mystep)))

for ii in range(len(n_positions)):
    for jj in range((len(mystep))):
        if n_positions[ii] == 4 and mystep[jj] == 90:
            print("Hallo")
        print("n_positions = %s\n" % n_positions[ii])
        print("mystep = %s\n" % mystep[jj])
        delta_theta = 2 * math.pi / n_positions[ii]
        theta_matrix = np.array([np.arange(360, step=mystep[jj]) * math.pi / 180, ] * n_positions[ii])
        theta_shifts = np.array([np.arange(n_positions[ii]) * delta_theta, ] * theta_matrix.shape[1]).transpose()
        theta_response = theta_matrix - theta_shifts - theta_m
        matrix_A = np.power(mic_resp['cardioid'](theta_response), 2)
        store_rank[ii,jj] = np.linalg.matrix_rank(matrix_A)
        # eig_matrix[ii,jj], eigv_matrix[ii,jj] = np.linalg.eig(matrix_A.T @ matrix_A)
        print("matrix size = %s" % matrix_A.size)

np.save("rank_stored.npy", eig_matrix)

plt.imshow(store_rank)
plt.show()



