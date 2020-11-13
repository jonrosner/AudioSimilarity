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


sample_rate = 1 / 44100
num_samples = 200_000

x = np.linspace(0, num_samples * sample_rate, num_samples)
y = np.sin(500 * 2 * np.pi * x)
y += np.sin(2000 * 2 * np.pi * x)
yf = fft(y, norm='ortho')
yf = np.abs(yf[0:20_000:20])
xf = np.linspace(0.0, 1.0/(2.0*sample_rate), num_samples//2)[:20_000:20]

fig, (ax1, ax2) = plt.subplots(figsize=(9, 4), ncols=1, nrows=2)
fig.tight_layout(pad=3)
fig.set_size_inches(w=6.202, h=3)

ax1.set_title('Time Domain')
ax1.set(xlabel='Time', ylabel='Amplitude')

ax2.set_title('Frequency Domain')
ax2.set(xlabel='Frequency', ylabel='Magnitude')

ax1.plot(x[:1000], y[:1000])
ax2.plot(xf, yf)
plt.savefig('histogram.pgf')


