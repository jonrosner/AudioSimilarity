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
import pyrubberband as pyrb

t = np.linspace(0, 0.1, 16_000, False)
y = np.sin(2*np.pi*100*t)

t *= 1000

indices = np.round(np.arange(0, len(y), 2))
indices = indices[indices < len(y)].astype(int)

y_resampled = y[indices]

plt.plot(t, y)
plt.plot(t[:len(y_resampled)], y_resampled, color="red")

plt.gca().set(xlabel='Time (ms)', ylabel='Amplitude')

fig = plt.gcf()
fig.set_size_inches(w=6.202, h=2)
plt.tight_layout()

#plt.show()
plt.savefig("resampling.pgf")

