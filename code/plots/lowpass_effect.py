import matplotlib

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

import matplotlib.pyplot as plt
from scipy.fft import fft
import numpy as np
from scipy import signal

sample_rate = 1 / 44100
num_samples = 200_000

t = np.linspace(0, num_samples * sample_rate, num_samples)
y = np.sin(2*np.pi*500*t) + np.sin(2*np.pi*2000*t)
yf = fft(y, norm='ortho')
yf = np.abs(yf[0:20_000:20])
xf = np.linspace(0.0, 1.0/(2.0*sample_rate), num_samples//2)[:20_000:20]

sos = signal.butter(10, 15, 'lp', fs=1000, output='sos')
filtered = signal.sosfilt(sos, y)

fft_filtered = fft(filtered, norm='ortho')
fft_filtered = np.abs(fft_filtered[0:20_000:20])

fig, (ax1,ax2) = plt.subplots(ncols=1, nrows=2)

ax1.plot(t[:441] * 1000, y[441:441*2])
ax1.plot(t[:441] * 1000, filtered[441:441*2], color="red")
ax1.set_xlabel("Time (ms)")
ax1.set_ylabel("Amplitude")

ax2.plot(xf, yf)
ax2.plot(xf, fft_filtered, color="red")
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Mangitude")

fig = plt.gcf()
fig.set_size_inches(w=6.202, h=3.5)
plt.tight_layout()

#plt.show()
plt.savefig("lowpass_effect.pgf")