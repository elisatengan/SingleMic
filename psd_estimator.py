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

def estimate_psd(signal, fs, method='periodogram', window='hanning', nperseg=256, noverlap=None):
    freq = None
    signal_psd = None
    if noverlap is None:
        noverlap = round(nperseg/2)

    if method == 'periodogram':
        signal_FT = np.fft.fft(signal,len(signal))/len(signal)
        freq = np.fft.fftfreq(len(signal))*fs
        signal_psd = np.power(abs(signal_FT),2)/(fs/len(signal))

    if method == 'bartlett':
        nsegs = round(len(signal) / nperseg)
        seg_FT = np.empty((nperseg, nsegs), dtype=complex)
        for i in range(nsegs):
            segment = signal[i * nperseg:(i + 1) * nperseg]
            seg_FT[:, i] = np.fft.fft(segment, nperseg)/nperseg
        seg_psd = np.power(abs(seg_FT), 2)/(fs/nperseg)
        freq = np.fft.fftfreq(nperseg)*fs
        signal_psd = np.mean(seg_psd,axis=1)

    if method == 'welch':
        hann_window = scipy.signal.get_window(window, nperseg)

        S1 = (abs(sum(hann_window))) ** 2

        S2 = (hann_window * hann_window).sum()
        ENBW = fs * (S2 / S1)

        nwindows = int(round(len(signal) - noverlap) / (nperseg - noverlap))
        noffset = nperseg - noverlap
        seg_FT = np.empty((nperseg, nwindows), dtype=complex)
        sig = scipy.signal.detrend(signal, type='constant')
        for i in range(nwindows):
            segment = np.multiply(sig[i * noffset:(i * noffset) + nperseg], hann_window)
            seg_FT[:, i] = (np.fft.fft(segment, nperseg)) / nperseg
        seg_psd = (np.power(abs(seg_FT), 2)) / ENBW
        freq = np.fft.fftfreq(nperseg) * fs
        signal_psd = np.mean(seg_psd, axis=1)
        # freq, signal_psd = scipy.signal.welch(signal, fs=fs, nperseg=nperseg, window=window, noverlap=noverlap)

    return freq, signal_psd


def main():
    print("Hello psd_estimator.py")


if __name__ == '__main__':
    main()
