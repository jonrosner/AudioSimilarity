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

t = np.linspace(0, 0.1, 500, False)
y = np.sin(2*np.pi*100*t)
wn = np.clip((np.random.rand(len(y)) * 2 - 1) * 0.3, -1, 1)
y_wn = np.clip(y + wn, -1, 1)

t *= 1000

plt.plot(t, y_wn, color="red")
plt.plot(t, y)

plt.gca().set(xlabel='Time (ms)', ylabel='Amplitude')

fig = plt.gcf()
fig.set_size_inches(w=6.202, h=2)
plt.tight_layout()

#plt.show()
plt.savefig("whitenoise.pgf")
