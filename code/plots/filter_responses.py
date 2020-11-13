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


b,a = signal.butter(4, 200, 'low', analog=True)
t, y = signal.impulse([b,a])
t_step, y_step = signal.step([b,a])
w, h = signal.freqs(b, a)

fig, (ax1,ax2) = plt.subplots(ncols=1, nrows=2)

ax1.plot(t,y)
ax1.set_title('Impulse response')
ax1.set(xlabel='Time (s)', ylabel='Amplitude')
ax1.grid(which='both', axis='both')
ax1.margins(x=0)

ax2.semilogx(w, 20 * np.log10(abs(h)))
ax2.axvline(100, color='green')
ax2.set_title('Frequency response')
ax2.set(xlabel='Frequency (Hz)', ylabel='Magnitude (dB)')
ax2.grid(which='both', axis='both')
ax2.margins(x=0)

fig.set_size_inches(w=6.202, h=3)
fig.tight_layout()

#plt.show()
plt.savefig('filter_responses.pgf')
