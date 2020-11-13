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

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

window = np.hamming(51)
plt.plot(window)

plt.gca().axes.get_yaxis().set_visible(False)
plt.gca().axes.get_xaxis().set_visible(False)
plt.gca().set_axis_off()
plt.margins(0.1,0.1)
fig = plt.gcf()
fig.set_size_inches(3,0.5)
#plt.show()
plt.savefig("window_function.pgf", pad_inches=0, transparent=True)