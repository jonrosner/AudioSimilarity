import matplotlib

import seaborn as sns

# sns.set(rc={
#  "axes.axisbelow": False,
#  "axes.edgecolor": "lightgrey",
#  "axes.facecolor": "None",
#  "axes.grid": False,
#  "axes.spines.right": False,
#  "axes.spines.top": False,
#  "figure.facecolor": "white",
#  "lines.solid_capstyle": "round",
#  "patch.edgecolor": "w",
#  "patch.force_edgecolor": True,
#  "xtick.bottom": False,
#  "xtick.direction": "out",
#  "xtick.top": False,
#  "ytick.direction": "out",
#  "ytick.left": False,
#  "ytick.right": False,
#  "legend.facecolor": "white",
#  "legend.framealpha": 1,
#  "legend.loc": "upper right"})

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# cs = {
#     "red": (0.91, 0.604, 0.604),
#     "yellow": (0.95686275, 0.76078431, 0.05098039),
#     "blue": (0.639, 0.761, 0.973),
#     "green": (0.655, 0.863, 0.659),
#     "orange": (0.941, 0.741, 0.533),
#     "purple": (0.773, 0.682, 0.784)
# }

cs = {
    "yellow": (0.95686275, 0.76078431, 0.05098039),
    "red": (0.85882353, 0.19607843, 0.21176471),
    "blue": (0.28235294, 0.52156863, 0.92941176),
    "green": (0.23529412, 0.72941176, 0.32941176),
    "orange": (0.90196078, 0.47843137, 0.08627451),
    "purple": (0.56862745, 0.35686275, 0.56862745),
}
ax = plt.gca()
ax.axvline(200, color='grey', ls="--", alpha=0.3)

for i in range(6,1,-1):
    b,a = signal.butter(i, 200, fs=16000)
    w, h = signal.freqz(b, a, fs=16000)
    plt.plot(w, abs(h), color=cs[list(cs.keys())[i-1]])

ax.set_xlim((50, 600))
ax.set(xlabel='Frequency (Hz)', ylabel='Amplitude')
plt.gcf().set_size_inches(w=3.5, h=2.5)
plt.tight_layout()

#plt.show()
plt.savefig("butter_order.pgf")