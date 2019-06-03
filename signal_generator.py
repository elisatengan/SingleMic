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

"""
Defining the ideal microphone response patterns
"""


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
Function for generating complex exponential signals with random phase.

Parameters
-------------
L : Number of sources
fs : sampling frequency in Hz
f0 : fundamental frequency of complex exponential signals. All signals here have the same frequency
duration : signal duration in seconds
positions : source positions. 3 options are possible:
            1. 'uniform' -> sources are uniformly distributed along the azimuthal plane
            2. 'random' -> the source positions are randomly sorted
            3. numpy ndarray -> the source positions are passed as an array. If the number L and the 
            length of the array don't match, the function quits
amplitudes : array with amplitudes for signals. If not passed as an argument, all amps are set to 1.
            If the number L and the length of the array don't match, the function quits

Returns
------------
theta_sources : source positions in radians
sources_sig : L x (duration*fs) ndarray with each source signal in a row    

"""


def generate_sources(L, fs, f0, duration, positions, amplitudes=None):

    # Setting up the source positions
    if positions=='uniform':
        theta_sources = np.linspace(0, 2 * math.pi * (1 - 1 / L), L)
    if positions=='random':
        theta_sources = np.random.uniform(low=0., high=2 * math.pi, size=(L,))

    # Verifying number of position elements
    if isinstance(positions,np.ndarray):
        if len(positions)!=L:
            print("Number of position elements doesn't match number of sources")
            return
    # Setting up signals' amplitudes
    if amplitudes is None:
        amp_array = np.ones((L,))
    else:
        if len(amplitudes) != L:
            print("Number of amplitude elements doesn't match number of sources")
            return
        else:
            amp_array = amplitudes

    Nsec = duration  # time window in seconds
    N = round(Nsec * fs)  # number of samples
    tstep = (Nsec / N)  # time step between each sample
    time = np.arange(N) * tstep
    omega0 = 2*math.pi*f0

    # Sorting random phases for signals
    phi_array = np.random.uniform(low=0., high=2*math.pi, size=(L,))

    # Generating signals
    sources_sig = np.empty((L, len(time)), dtype=complex)
    for i in range(L):
        sources_sig[i, :] = amp_array[i] * np.exp(1j * (omega0 * time + phi_array[i]))

    return theta_sources, sources_sig


"""
Function for generating microphone's output signal.

Parameters
-------------
sources_sig: numpy ndarray with each signal in each row
theta_sources : source positions in Hz
npos : number of different positions in which the microphone records. Microphone rotates
       with a constant angular shift.
micresp_string : string with name of microphone response to be considered
theta_m : microphone's initial position relative to "global" reference


Returns
------------
nsamples_seg : number of samples obtained per microphone position
delta_theta : angular distance between microphone positions
mic_output :  1D array with microphone's output signal

-------------------------------------------------------------------------------------------------
WARNING : This function is not yet robust enough. It still considers that the user will be smart
          enough to choose a signal duration that will lead to na integer number of samples
          per positions considered
-------------------------------------------------------------------------------------------------

"""


def generate_mic_output(sources_sig, theta_sources,npos, micresp_string, theta_m=0.):

    # Getting the number of sources
    L = np.shape(sources_sig)[0]
    # Calculating the number of samples for each mic position
    nsamples_seg = int(round((np.shape(sources_sig)[1])/npos))

    mic_sigs = np.empty(np.shape(sources_sig), dtype=complex)

    # Microphone's angular distance between consecutive positions
    delta_theta = (2*math.pi/npos)

    # Calculating mic's actual positions
    positions = (np.arange(npos) * delta_theta) + theta_m

    # Computing auxiliary matrix for generating output
    mic_positions_matrix = np.array([positions, ] * L).transpose()
    sources_positions_matrix = np.ones((npos, L)) * theta_sources
    theta_relative = sources_positions_matrix - mic_positions_matrix

    # Generating output
    for i in range(npos):
        for k in range(L):
            # if i == npos-1:
            #     mic_sigs[k, i*nsamples_seg:-1] = mic_resp[micresp_string](theta_relative[i, k])\
            #                                                  * sources_sig[k, i*nsamples_seg:-1]
            # else:
            mic_sigs[k, i*nsamples_seg:(i+1)*nsamples_seg] = mic_resp[micresp_string](theta_relative[i, k])\
                                                             * sources_sig[k, i*nsamples_seg:(i+1)*nsamples_seg]
            # TODO: make it more robust to different lengths - n_pos combinations

    mic_output = sum(mic_sigs, 0)

    return nsamples_seg, delta_theta, mic_output


def main():

    print("Hello signal_generator")


if __name__ == '__main__':
    main()



