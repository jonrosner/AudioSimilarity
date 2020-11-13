import matplotlib

import seaborn as sns

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

import matplotlib.pyplot as plt
import pyrubberband as pyrb
from scipy import signal
import librosa
import numpy as np

y, _ = librosa.core.load("/home/joro/uni/master-thesis/dataset/testset-songs/2mRNUz2RbAFf0AkPeQWelt", mono=True, sr=44100, dtype=np.float32)

print(max(y), min(y))
y = y[:3*44100]
wn = np.clip((np.random.rand(len(y)) * 2 - 1) * 0.01, -1, 1)
y_wn = np.clip(y + wn, -1, 1)

gain_float = gain = 10 ** (6 / 20)
y_gain = np.clip(y * gain_float, -1, 1)

B, A = signal.butter(5, 5000 / (44100 / 2), btype='lowpass')
y_lp = signal.lfilter(B, A, y, axis=0)

B, A = signal.butter(5, 10000 / (44100 / 2), btype='highpass')
y_hp = signal.lfilter(B, A, y, axis=0)

y_stretch = pyrb.time_stretch(y, 44100, 0.7)[:3*44100]

y_pitch = pyrb.pitch_shift(y, 44100, -5)

fig, (ax1, ax2, ax3) = plt.subplots(ncols=2, nrows=3)
fig.tight_layout(pad=3)
Pxx, freqs, bins, im = ax1[0].specgram(y, NFFT=1024, Fs=44100, noverlap=900, scale_by_freq=True, detrend="mean")
ax1[0].set_title('Original')
ax1[0].set(ylabel='Frequency (Hz)')
ax1[0].xaxis.set_ticks([])
ax1[0].yaxis.set_ticks([0, 10_000, 20_000])
ax1[0].yaxis.set_ticklabels(["0", "10k", "20k"])
Pxx, freqs, bins, im = ax1[1].specgram(y_wn, NFFT=1024, Fs=44100, noverlap=0, scale_by_freq=True, detrend="mean")
ax1[1].set_title('Whitenoise')
ax1[1].xaxis.set_ticks([])
ax1[1].yaxis.set_ticks([])

Pxx, freqs, bins, im = ax2[0].specgram(y_hp, NFFT=1024, Fs=44100, noverlap=0, scale_by_freq=True, detrend="mean")
ax2[0].set_title('Highpass Filter')
ax2[0].set(ylabel='Frequency (Hz)')
ax2[0].xaxis.set_ticks([])
ax2[0].yaxis.set_ticks([0, 10_000, 20_000])
ax2[0].yaxis.set_ticklabels(["0", "10k", "20k"])
Pxx, freqs, bins, im = ax2[1].specgram(y_lp, NFFT=1024, Fs=44100, noverlap=0, scale_by_freq=True, detrend="mean")
ax2[1].set_title('Lowpass Filter')
ax2[1].xaxis.set_ticks([])
ax2[1].yaxis.set_ticks([])

Pxx, freqs, bins, im = ax3[0].specgram(y_stretch, NFFT=1024, Fs=44100, noverlap=0, scale_by_freq=True, detrend="mean")
ax3[0].set_title('Timestretch')
ax3[0].set(xlabel='Time (s)', ylabel='Frequency (Hz)')
ax3[0].xaxis.set_ticks([0, 1, 2, 3])
ax3[0].yaxis.set_ticks([0, 10_000, 20_000])
ax3[0].yaxis.set_ticklabels(["0", "10k", "20k"])
Pxx, freqs, bins, im = ax3[1].specgram(y_pitch, NFFT=1024, Fs=44100, noverlap=0, scale_by_freq=True, detrend="mean")
ax3[1].set_title('Pitchshift')
ax3[1].set(xlabel='Time (s)')
ax3[1].xaxis.set_ticks([0, 1, 2, 3])
ax3[1].yaxis.set_ticks([])

fig = plt.gcf()
fig.set_size_inches(w=6.202, h=5)
plt.tight_layout()

#plt.show()
plt.savefig("augmentations.pgf")