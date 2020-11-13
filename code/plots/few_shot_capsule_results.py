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

import matplotlib.pyplot as plt
import numpy as np

cs = {
    "red": (0.85882353, 0.19607843, 0.21176471),
    "yellow": (0.95686275, 0.76078431, 0.05098039),
    "blue": (0.28235294, 0.52156863, 0.92941176),
    "green": (0.23529412, 0.72941176, 0.32941176),
    "orange": (0.90196078, 0.47843137, 0.08627451),
    "purple": (0.56862745, 0.35686275, 0.56862745)
}

x = [10,20,40,60,80]
a = np.arange(len(x))
y_ticks = [10,20,30,40,50,60,70]
y_tickslabels = ["10.00","20.00","30.00","40.00","50.00","60.00","70.00"]
y_supervised = [14,26,48,55,61]
y_claudio = [37,50,60,66,68]
y_capsule = [28,40,53,58,59]
y_resnet = [23,41,59,66,69]

plt.hlines(y=[10,20,30,40,50,60,70], xmin=-1, xmax=5, linestyles='--', lw=1, color="lightgrey", zorder=1)

plt.plot(a, y_supervised, "o-", color=cs["green"], zorder=2)
plt.plot(a, y_claudio, "o-", color=cs["red"], zorder=2)
plt.plot(a, y_capsule, "o-", color=cs["orange"], zorder=2)
#plt.plot(a, y_resnet, "o--", color="blue", zorder=2, alpha=0.3)

ax = plt.gca()
ax.set_xlim(-0.5,4.5)
ax.xaxis.set_ticks(a)
ax.xaxis.set_ticklabels(x)
ax.yaxis.set_ticks(y_ticks)
ax.yaxis.set_ticklabels(y_tickslabels)
ax.set(xlabel="Samples per Class", ylabel="Top-1 Accuracy (%)")

fig = plt.gcf()
fig.set_size_inches(w=6.202, h=4)
plt.tight_layout()

plt.legend(["Supervised VGG", "CL-Audio VGG", "CapsuleNet-M"])

#plt.show()
plt.savefig("few_shot_capsule_result.pgf")