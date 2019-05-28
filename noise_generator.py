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


def generate_noise(L, fs, duration,positions, noise_power):
    if positions=='uniform':
        theta_sources = np.linspace(0, 2 * math.pi * (1 - 1 / L), L)
    if positions=='random':
        theta_sources = np.random.uniform(low=0., high=2 * math.pi, size=(L,))
    if isinstance(positions,np.ndarray):
        if len(positions)!=L:
            print("Wrong array")
            return

    if len(noise_power) != L:
        print("number of noise power elements doesnt match number of sources")

    Nsec = duration  # time window in seconds
    N = round(Nsec * fs)  # number of samples
    tstep = (Nsec / N)  # time step between each sample
    time = np.arange(N) * tstep
    amp_array = np.ones((L,))
    sources_sig = np.empty((L,len(time)),dtype=complex)

    for i in range(L):
        sources_sig[i,:] = np.random.normal(scale=np.sqrt(noise_power[i]*fs/2),size=N)

    return theta_sources, sources_sig


def main():
    fs = 16000

    print("Hello noise_generator")


if __name__ == '__main__':
    main()


