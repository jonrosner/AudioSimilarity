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

t = np.linspace(0, 10, 10*16_000, False)
y = np.sin(2*np.pi*1*t)

#t *= 1000

plt.plot(t, y)
plt.plot(t[3*16_000:6*16_000], y[3*16_000:6*16_000], color="red")

ax = plt.gca()
ax.axvline(3, color='red', ls="--")
ax.axvline(6, color='red', ls="--")

plt.gca().set(xlabel='Time (s)', ylabel='Amplitude')

fig = plt.gcf()
fig.set_size_inches(w=6.202, h=2)
plt.tight_layout()

#plt.show()
plt.savefig("crop.pgf")
