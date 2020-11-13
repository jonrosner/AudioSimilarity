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

fig, ax = plt.subplots(ncols=2)
fig.set_size_inches(w=6.202, h=2)

x = [0,4,6,10]
y_lowpass = [1,1,0,0]

ax[0].plot(x,y_lowpass, "-")
ax[0].vlines([4,6], -1, 2, linestyles="dashed")
ax[0].set_ylim(0, 1.1)
ax[0].axes.yaxis.set_ticks([])
ax[0].axes.xaxis.set_ticks([])
ax[0].set(xlabel='Frequency', ylabel='Amplitude')
ax[0].set_title('Low-pass Filter')
ax[0].text(0, 0.85, "Passband")
ax[0].text(6.5, 0.06, "Stopband")
ax[0].annotate("Transition\nband", xy=(5,0.5), xycoords="data", xytext=(0.8,0.7), textcoords="axes fraction",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
            horizontalalignment='center', verticalalignment='top')

y_highpass = [0,0,1,1]

ax[1].plot(x,y_highpass, "-")
ax[1].set_ylim(0, 1.1)
ax[1].axes.yaxis.set_ticks([])
ax[1].axes.xaxis.set_ticks([])
ax[1].set(xlabel='Frequency', ylabel='Amplitude')
ax[1].set_title('High-pass Filter')

fig.tight_layout()
# plt.show()
plt.savefig('lowpass_highpass.pgf')