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

filepath = "/home/joro/Downloads/wav/id10270/5r0dWxy17C8/00001.wav"
sr, y = wavfile.read(filepath)
y = y / 2**15
x = np.linspace(0, len(y) / sr, len(y))

plt.plot(x, y)
plt.gca().axes.get_yaxis().set_visible(False)
plt.gca().axes.get_xaxis().set_visible(False)
plt.gca().set_axis_off()
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
plt.margins(0,0)
fig = plt.gcf()
fig.set_size_inches(w=4, h=1)
plt.show()
plt.savefig("waveform_claudio.pgf", pad_inches=0, transparent=True)