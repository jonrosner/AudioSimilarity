import matplotlib

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy import signal

with open("signal3.npx", "rb") as f:
    y = np.load(f)

Pxx, freqs, bins, im = plt.specgram(y, NFFT=1024, Fs=44100, noverlap=900, scale_by_freq=True, detrend="mean")

plt.gca().axes.get_yaxis().set_visible(False)
plt.gca().axes.get_xaxis().set_visible(False)
plt.gca().set_axis_off()
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
plt.margins(0,0)
fig = plt.gcf()
fig.set_size_inches(w=2, h=1)
#plt.show()
plt.savefig("spectrogram_augmented3.pgf", pad_inches=0, transparent=True)