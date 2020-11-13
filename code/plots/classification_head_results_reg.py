import matplotlib

# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False,
# })

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.ndimage.filters import gaussian_filter1d


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
 "text.color": "dimgrey",
 "xtick.bottom": False,
 "xtick.color": "dimgrey",
 "xtick.direction": "out",
 "xtick.top": False,
 "ytick.color": "dimgrey",
 "ytick.direction": "out",
 "ytick.left": False,
 "ytick.right": False})

y_ticks = [25,30,40,50]
y_tickslabels = ["25.00","30.00","40.00","50.00"]

x_ticks = [0, 500_000, 1_000_000, 1_500_000, 2_000_000]
x_tickslabels = ["0","500k","1M","1.5M", "2.0M"]

layer1 = [(1104946, 41.22), (553522, 43.35), (277811, 50.22), (139954, 46.97, "[128]"), (71026, 45.84, "[64]"), (36562, 33.21, "[32]"), (19331, 27.74, "[16]"), (10714, 25.93, "[8]")]
layer2 = [(2156594, 44.94, "[1024,1024]"), (1605170, 46.51, "[1024,512]"), (423218, 47.01, "[256,512]"), (344114, 45.94, "[256,256]"), (564082, 40.72, "[512,64]"), (145138, 40.41, "[128,64]")]
layer3 = [(1724210, 42.58, "[1024,512,256]"), (739314, 43.19, "[512,256,128]"), (469938, 42.11, "[256,512,128]")]

layers = layer1 + layer2 + layer3
layers = sorted(layers, key=lambda t: t[0])
x = [l[0] for l in layers]
y = [l[1] for l in layers]
y_smoothed = gaussian_filter1d(y, sigma=1.5)

plt.plot(x, y_smoothed, color="blue")

ax = plt.gca()

ax.yaxis.set_ticks(y_ticks)
ax.yaxis.set_ticklabels(y_tickslabels)
ax.xaxis.set_ticks(x_ticks)
ax.xaxis.set_ticklabels(x_tickslabels)
ax.set(xlabel="Num. of trainable parameters", ylabel="Top-1 Accuracy (%)")

fig = plt.gcf()
fig.set_size_inches(w=6.202, h=6.202)
plt.tight_layout()

plt.legend(["1-Layer", "2-Layers", "3-Layers"])

plt.show()
#plt.savefig("few_shot_capsule_result.pgf")