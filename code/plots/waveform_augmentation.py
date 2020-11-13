import matplotlib

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy import signal

filepath = "/home/joro/Downloads/wav/id10270/5r0dWxy17C8/00001.wav"
sr, y = wavfile.read(filepath)
y = y / 2**15
y = y[2:sr*5]
# wn = np.clip((np.random.rand(len(y)) * 2 - 1) * 0.01, -1, 1)
# y = np.clip(y + wn, -1, 1)
# sos = signal.butter(5, 300, 'hp', fs=sr, output='sos')
# y = signal.sosfilt(sos, y)
x = np.linspace(0, len(y) / sr, len(y))

plt.plot(x, y)
plt.gca().axes.get_yaxis().set_visible(False)
plt.gca().axes.get_xaxis().set_visible(False)
plt.gca().set_axis_off()
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
plt.margins(0,0)
fig = plt.gcf()
fig.set_size_inches(w=2, h=1)
with open("signal3.npx", "wb+") as f:
    np.save(f,y)
# plt.show()
plt.savefig("waveform_augmented3.pgf", pad_inches=0, transparent=True)