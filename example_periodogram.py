from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1234)

fs = 16000
N = 1e5
amp = 2*np.sqrt(2)
freq = 1600.0
noise_power = 0.001 * fs / 2
time = np.arange(N) / fs
x = amp*np.sin(2*np.pi*freq*time)
x += np.random.normal(scale=np.sqrt(noise_power), size=time.shape)


f, Pxx_spec = signal.periodogram(x, fs, 'boxcar', scaling='spectrum', return_onesided=True)
plt.figure()
plt.semilogy(f, Pxx_spec)
plt.ylim([1e-8, 1e1])
plt.xlim(0, fs/2)
plt.xlabel('frequency [Hz]')
plt.ylabel('Power spectrum [V**2 RMS]')


dft = np.fft.fft(x, len(x))
Pxx_elisa = np.power(np.abs(dft), 2)/(len(x)**2)
plt.figure()
plt.semilogy(Pxx_elisa)
plt.ylabel('Linear spectrum [V^2 RMS]')

y = np.fft.fft(x, len(x))
freqs = np.fft.fftfreq(len(x))
plt.figure()
plt.plot(freqs*fs, np.abs(y)/len(y))
plt.xlabel("frequency [Hz]")
plt.ylabel(r"Magnitude")
plt.xlim(0, fs/2)
plt.show()




