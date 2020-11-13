import matplotlib

import seaborn as sns

sns.set(rc={
 "axes.axisbelow": False,
 "axes.edgecolor": "lightgrey",
 "axes.facecolor": "None",
 "axes.grid": False,
 "axes.spines.right": False,
 "axes.spines.top": False,
 "figure.facecolor": "white",
 "lines.solid_capstyle": "round",
 "patch.edgecolor": "w",
 "patch.force_edgecolor": True,
 "xtick.bottom": False,
 "xtick.direction": "out",
 "xtick.top": False,
 "ytick.direction": "out",
 "ytick.left": False,
 "ytick.right": False,
 "legend.facecolor": "white"})

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

cs = {
    "red": (0.85882353, 0.19607843, 0.21176471),
    "yellow": (0.95686275, 0.76078431, 0.05098039),
    "blue": (0.28235294, 0.52156863, 0.92941176),
    "green": (0.23529412, 0.72941176, 0.32941176),
    "orange": (0.90196078, 0.47843137, 0.08627451),
    "purple": (0.56862745, 0.35686275, 0.56862745)
}

import matplotlib.pyplot as plt
import numpy as np

seconds = [861, 1041, 798]

y_ticks = [120 * i for i in range(10)]
y_tickslabels = [int(i / 60) for i in y_ticks]

ind = np.arange(len(seconds))  # the x locations for the groups
width = 0.5  # the width of the bars

rect1 = plt.bar(0, seconds[0], width, align="center",
                color=cs["red"], label='Men', zorder=2)
rect2 = plt.bar(1, seconds[1], width, align="center",
                color=cs["green"], label='Men', zorder=2)
rect3 = plt.bar(2, seconds[2], width, align="center",
                color=cs["blue"], label='Men', zorder=2)

plt.hlines(y=y_ticks, xmin=-1, xmax=5, linestyles='--', lw=1, color="lightgrey", zorder=1, label='_nolegend_')

ax = plt.gca()
ax.set_xticks(ind)
ax.set_xticklabels(("VGG-Vox", "LSTM", "SpeechTransformer"))
ax.yaxis.set_ticks(y_ticks)
ax.yaxis.set_ticklabels(y_tickslabels)
ax.set(ylabel="Epoch Time (m)")
ax.set(xlabel="Encoder Architecture")
ax.set_xlim(-0.5,2.5)

fig = plt.gcf()
fig.set_size_inches(w=5, h=4)
plt.tight_layout()

#plt.show()
plt.savefig("time_per_epoch.pgf")