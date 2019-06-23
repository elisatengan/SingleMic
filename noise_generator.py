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


def generate_noise(L, fs, duration, positions, noise_power):

    """
    Function for generating noise signals.

    Parameters
    -------------
    L : number of sources
    fs : sampling frequency in Hz
    duration : signal duration in seconds
    positions : source positions. 3 options are possible:
                1. 'uniform' -> sources are uniformly distributed along the azimuthal plane
                2. 'random' -> the source positions are randomly sorted
                3. numpy ndarray -> the source positions are passed as an array. If the number L and the
                length of the array don't match, the function quits
    noise_power : array with noise power of each source. If the number L and the length of the array don't match,
    the function quits

    Returns
    ------------
    theta_sources : source positions in radians
    sources_sig : L x (duration*fs) ndarray with each source signal in a row

    """

    # Setting up the source positions
    if isinstance(positions,np.ndarray):
        theta_sources = positions
    elif positions=='uniform':
        theta_sources = np.linspace(0, 2 * math.pi * (1 - 1 / L), L)

    elif positions=='random':
        theta_sources = np.random.uniform(low=0., high=2 * math.pi, size=(L,))

    # Verifying number of position elements
    if isinstance(positions,np.ndarray):
        if len(positions)!=L:
            print("Number of position elements doesn't match number of sources")
            return

    # Verifying number of noise power elements
    if len(noise_power) != L:
        print("Number of noise power elements doesn't match number of sources")
        return

    Nsec = duration  # time window in seconds
    N = round(Nsec * fs)  # number of samples
    tstep = (Nsec / N)  # time step between each sample
    time = np.arange(N) * tstep
    sources_sig = np.empty((L, len(time)), dtype=complex)

    # Generating signals
    for i in range(L):
        sources_sig[i,:] = np.random.normal(scale=np.sqrt(noise_power[i]*fs/2), size=N)

    return theta_sources, sources_sig


def main():
    print("Hello noise_generator")


if __name__ == '__main__':
    main()


