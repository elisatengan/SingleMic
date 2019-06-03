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
import scipy.signal


def estimate_psd(signal, fs, method='periodogram', window='hanning', nperseg=256, noverlap=None):

    """
    Function for estimating the power spectral density of a time domain signal.

    Parameters
    --------------
    signal : time domain signal
    fs : sampling frequency in Hz
    method : PSD estimation method:
             -'periodogram'
             -'bartlett'
             -'welch'
    window : Window to be used in the case of Welch's PSD estimation method.
             Options are the ones accepted by the scipy.signal function "get_window"
    nperseg : number of samples per window. Default is 256
    noverlap : number of overlapping samples between windows. Default is nperseg/2

    Returns
    --------------
    freq : 1D array with frequency values for estimated PSD
    signal_psd : 1D array with PSD estimation for each freq value

    -------------------------------------------------------------------------------------------------
    WARNING : This function is not yet robust enough. It still considers that the user will be smart
              enough to have a case where there is an integer number of windows "fitting" inside
              the whole signal - that will depend on the length of the time domain signal and the
              parameters nperseg and noverlap
    -------------------------------------------------------------------------------------------------


    """

# TODO: Make it more robust to different signal/window lengths and overlaps

    freq = None
    signal_psd = None
    if noverlap is None:
        noverlap = round(nperseg/2)

    # Periodogram implementation
    if method == 'periodogram':
        signal_FT = np.fft.fft(signal, len(signal))/len(signal)
        freq = np.fft.fftfreq(len(signal))*fs
        signal_psd = np.power(abs(signal_FT), 2)/(fs/len(signal))

    # Bartlett implementation
    elif method == 'bartlett':
        nsegs = round(len(signal) / nperseg)
        seg_FT = np.empty((nperseg, nsegs), dtype=complex)

        for i in range(nsegs):
            segment = signal[i * nperseg:(i + 1) * nperseg]
            seg_FT[:, i] = np.fft.fft(segment, nperseg)/nperseg

        seg_psd = np.power(abs(seg_FT), 2)/(fs/nperseg)
        freq = np.fft.fftfreq(nperseg)*fs
        signal_psd = np.mean(seg_psd, axis=1)

    # Welch implementation
    elif method == 'welch':
        my_window = scipy.signal.get_window(window, nperseg)

        S1 = (abs(sum(my_window))) ** 2

        nwindows = int(round((len(signal) - noverlap) / (nperseg - noverlap)))
        noffset = nperseg - noverlap
        seg_FT = np.empty((nperseg, nwindows), dtype=complex)
        sig = scipy.signal.detrend(signal, type='constant')  # is this necessary?

        for i in range(nwindows):
            segment = np.multiply(sig[i * noffset:(i * noffset) + nperseg], my_window)
            seg_FT[:, i] = (np.fft.fft(segment, nperseg)) / nperseg

        seg_psd = (np.power(abs(seg_FT), 2)) / S1
        freq = np.fft.fftfreq(nperseg) * fs
        # If for some reason there is only one window, no averaging is done (this adaptation was for a specific test)
        if seg_psd.shape[1] == 1:
            signal_psd = seg_psd.reshape((len(seg_psd),))
        # Averaging of PSD estimations
        else:
            signal_psd = np.mean(seg_psd, axis=1)

    else:
        print("Unknown estimation method (valid options: 'periodogram', 'bartlett', 'welch')")
        return

    return freq, signal_psd


def main():
    print("Hello psd_estimator.py")


if __name__ == '__main__':
    main()
