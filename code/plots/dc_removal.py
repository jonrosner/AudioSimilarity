
import matplotlib.pyplot as plt
from scipy.fft import fft
import numpy as np
from scipy import signal


w, h = signal.freqz([1,-1], [1,-0.99])
plt.plot(w, abs(h))
plt.xlim((0, 1))
plt.show()
