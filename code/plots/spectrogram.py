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
from scipy.io import wavfile

filepath = "/home/joro/Downloads/wav/id10270/5r0dWxy17C8/00001.wav"
sr, y = wavfile.read(filepath)
y = y[:63999]
y = y / 2**15
x = np.linspace(0, len(y) / sr, len(y))

fig, (ax1, ax2) = plt.subplots(ncols=1, nrows=2)

ax1.set(ylabel='Amplitude')
ax1.margins(0)
ax1.plot(x, y)

Pxx, freqs, bins, im = ax2.specgram(y, NFFT=1024, Fs=16000, noverlap=0, scale_by_freq=True, detrend="mean")
ax2.set(xlabel='Time (s)', ylabel='Frequency (Hz)')

fig = plt.gcf()
fig.set_size_inches(w=6.202, h=3.5)
plt.tight_layout()

plt.savefig('spectrogram.pgf')


