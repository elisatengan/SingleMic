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


def generate_sources(L, fs, f0, duration,positions):
    if positions=='uniform':
        theta_sources = np.linspace(0, 2 * math.pi * (1 - 1 / L), L)
    if positions=='random':
        theta_sources = np.random.uniform(low=0., high=2 * math.pi, size=(L,))
    if isinstance(positions,np.ndarray):
        if len(positions)!=L:
            print("Wrong array")
            return


    Nsec = duration  # time window in seconds
    N = round(Nsec * fs)  # number of samples
    tstep = (Nsec / N)  # time step between each sample
    time = np.arange(N) * tstep
    omega0 = 2*math.pi*f0
    amp_array = np.ones((L,))
    phi_array = np.random.uniform(low=0.,high=2*math.pi,size=(L,))
    sources_sig = np.empty((L,len(time)),dtype=complex)

    for i in range(L):
        if len(omega0) == 2:
            sources_sig[i,:] = amp_array[i] * np.exp(1j*(omega0[i]*time + phi_array[i]))
        else:
            sources_sig[i, :] = amp_array[i] * np.exp(1j * (omega0 * time + phi_array[i]))
    return theta_sources, sources_sig


def generate_mic_output(sources_sig, theta_sources,npos, micresp_string, theta_m=0.):
    L = np.shape(sources_sig)[0]
    nsamples_seg = round((np.shape(sources_sig)[1])/npos)
    mic_sigs = np.empty(np.shape(sources_sig),dtype=complex)
    delta_theta = (2*math.pi/npos)
    positions = (np.arange(npos) * delta_theta) + theta_m
    mic_positions_matrix = np.array([positions, ] * L).transpose()
    sources_positions_matrix = np.ones((npos, L)) * theta_sources
    theta_relative = sources_positions_matrix - mic_positions_matrix
    for i in range(npos):
        for k in range(L):
            # if i == npos-1:
            #     mic_sigs[k, i*nsamples_seg:-1] = mic_resp[micresp_string](theta_relative[i, k])\
            #                                                  * sources_sig[k, i*nsamples_seg:-1]
            # else:
            mic_sigs[k, i*nsamples_seg:(i+1)*nsamples_seg] = mic_resp[micresp_string](theta_relative[i, k])\
                                                             * sources_sig[k, i*nsamples_seg:(i+1)*nsamples_seg]

    mic_output = sum(mic_sigs, 0)
    return nsamples_seg, delta_theta, mic_output


def main():
    fs = 16000
    f0 = 200
    nb_sources = 2
    sources_signals=np.empty((2,96000),dtype=complex)
    angles_sources, sources_signals = generate_sources(nb_sources,fs,f0,6,'uniform')
    y = np.empty((96000,),dtype=complex)
    _,_,y = generate_mic_output(sources_signals,angles_sources,6,'cardioid', 0)
    fig = plt.figure()
    plt.plot(np.real(y))
    plt.plot(np.imag(y))

    print("Hello signal_generator")
    plt.show()

if __name__ == '__main__':
    main()



