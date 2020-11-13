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
gain_float = gain = 10 ** (6 / 20)
y_gain = np.clip(y * gain_float, -1, 1)

t *= 1000

plt.plot(t, y)
plt.plot(t, y_gain, color="red")

plt.gca().set(xlabel='Time (ms)', ylabel='Amplitude')

fig = plt.gcf()
fig.set_size_inches(w=6.202, h=2)
plt.tight_layout()

#plt.show()
plt.savefig("gain_2db.pgf")
