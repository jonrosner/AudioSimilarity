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

b,a = signal.butter(8, 200, fs=16000)
w_butter, h_butter = signal.freqz(b, a, fs=16000)

b,a = signal.ellip(4, 1, 20, 200, fs=16000)
w_ell, h_ell = signal.freqz(b, a, fs=16000)

b,a = signal.cheby1(8, 1, 200, fs=16000)
w_cheb1, h_cheb1 = signal.freqz(b, a, fs=16000)

b,a = signal.cheby2(8, 10, 200, fs=16000)
w_cheb2, h_cheb2 = signal.freqz(b, a, fs=16000)

fig, (ax1,ax2) = plt.subplots(ncols=2, nrows=2)

ax1[0].plot(w_butter, abs(h_butter))
ax1[1].plot(w_ell, abs(h_ell))
ax2[0].plot(w_cheb1, abs(h_cheb1))
ax2[1].plot(w_cheb2, abs(h_cheb2))


ax1[0].set_xlim((0, 400))
ax1[1].set_xlim((0, 400))
ax2[0].set_xlim((0, 400))
ax2[1].set_xlim((0, 400))

ax1[0].set(ylabel='Amplitude')
ax1[0].set_title('Butterworth')
ax1[1].set_title('Elliptic')
ax2[0].set(xlabel='Frequency', ylabel='Amplitude')
ax2[0].set_title('Chebyshev 1')
ax2[1].set(xlabel='Frequency')
ax2[1].set_title('Chebyshev 2')


fig = plt.gcf()
fig.set_size_inches(w=6.202, h=3)
plt.tight_layout()

#plt.show()
plt.savefig("lp_filters_compare.pgf")
