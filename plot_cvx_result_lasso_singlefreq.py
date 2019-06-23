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

import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt
import math

# Load data
filename = '200619_cvx_result_lasso_singlefreq_L=5.mat'
data = io.loadmat(filename)
nb_sources = int(filename[filename.find('L=')+2])

if nb_sources==1:
    theta = np.array([30*math.pi/180])
else:
    theta = np.linspace(0, 2 * math.pi * (1 - 1 / nb_sources), nb_sources)

x = data['x']
x_normalized = x/(max(x))
# Create angles for x-axis
mystep=1
L = int(360/mystep)
angles = np.arange(start=0, stop=360, step=mystep)
angles_rad = np.deg2rad(angles)

# Creating arrays of ticks and labels for a decent plot?
tick_step_degree = 30
tick_step_indices = int(tick_step_degree/mystep)
tick_array = np.arange(L, step=tick_step_indices)
label_array = np.empty((len(tick_array,)),dtype=object)
for i in range(len(label_array)):
    label_array[i] = '%s°' % (mystep*tick_array[i])

# Create figure

# fig = plt.figure(figsize=(16,6))
# plt.plot(angles.T, x)
# plt.xticks(ticks=tick_array,labels=label_array,fontsize=14)
# plt.yticks(fontsize=14)
# plt.ticklabel_format(axis='y',style='sci', scilimits=(0, 1))
# plt.axvline(x=tick_array[0],color='red',linestyle='--')
# plt.axvline(x=tick_array[6],color='red',linestyle='--')
# plt.xlabel('Position', fontsize=14)
# plt.ylabel("Coefficient", fontsize=14)
# plt.xlim(min(angles),max(angles))
#
# fig_polar = plt.figure()
# ax = plt.subplot(111, projection='polar')
# ax.stem(angles_rad,x)
# ax.spines['polar'].set_visible(False)
# ax = plt.gca()
# ax.set_ylim((min(x)-5e-6, max(x)+1e-6))
# ax.grid(axis='y')
# ax.set_yticks([])

fig_polar1 = plt.figure()
plt.polar(angles_rad,x_normalized,linewidth=3)
ax= plt.gca()
yticks = plt.yticks()
plt.xticks(tick_array*math.pi/180)
ax.set_ylim(0,1)
if nb_sources>1:
    for idx in range(len(theta)):
        ax.plot((0,theta[idx]),(0,1),'--r',linewidth=2)

else:
    ax.plot((0,theta),(0,1),'--r',linewidth=2)
# plt.yticks(ticks=plt.yticks()[0],labels=['']*len(plt.yticks()[0]))

plt.show()

