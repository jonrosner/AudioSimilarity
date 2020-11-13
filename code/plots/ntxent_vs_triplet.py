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

voxceleb100 = [72.15, 50.18]
voxceleb5020 = [50.22, 35.65]

y_ticks = [10,20,30,40,50,60,70]
y_tickslabels = ["10.00","20.00","30.00","40.00","50.00","60.00","70.00"]

ind = np.arange(len(voxceleb100))  # the x locations for the groups
width = 0.3  # the width of the bars

rects1 = plt.bar(ind - width/2, voxceleb100, width-0.05, align="center",
                color=cs["green"], label='Men', zorder=2)

rects2 = plt.bar(ind + width/2, voxceleb5020, width-0.05, align="center",
                color=cs["red"], label='Women', zorder=2)

plt.hlines(y=[10,20,30,40,50,60,70], xmin=-1, xmax=5, linestyles='--', lw=1, color="lightgrey", zorder=1, label='_nolegend_')

ax = plt.gca()
ax.set_xticks(ind)
ax.set_xticklabels(("VoxCeleb", "VoxCeleb50-20"))
ax.yaxis.set_ticks(y_ticks)
ax.yaxis.set_ticklabels(y_tickslabels)
ax.set(xlabel="Transfer Dataset", ylabel="Top-1 Accuracy (%)")
ax.set_xlim(-0.5,1.5)

plt.legend(["NT-Xent", "Triplet"])

fig = plt.gcf()
fig.set_size_inches(w=5, h=4)
plt.tight_layout()

#plt.show()
plt.savefig("ntxent_vs_triplet_result.pgf")