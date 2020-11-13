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

# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False,
# })

import matplotlib.pyplot as plt
import numpy as np

y_ticks = [25,30,40,50]
y_tickslabels = ["25.00","30.00","40.00","50.00"]

x_ticks = [0, 500_000, 1_000_000, 1_500_000, 2_000_000]
x_tickslabels = ["0","500k","1M","1.5M", "2.0M"]

layer1 = [(1104946, 41.22, "[1024]"), (553522, 43.35, "[512]"), (277811, 50.22, "[256]"), (139954, 46.97, "[128]"), (71026, 45.84, "[64]"), (36562, 33.21, "[32]"), (19331, 27.74, "[16]"), (10714, 25.93, "[8]")]
layer2 = [(2156594, 44.94, "[1024,1024]"), (1605170, 46.51, "[1024,512]"), (423218, 47.01, "[256,512]"), (344114, 45.94, "[256,256]"), (564082, 40.72, "[512,64]"), (145138, 40.41, "[128,64]")]
layer3 = [(1724210, 42.58, "[1024,512,256]"), (739314, 43.19, "[512,256,128]"), (469938, 42.11, "[256,512,128]")]

plt.scatter(x=[l[0] for l in layer1], y=[l[1] for l in layer1], color="green", s=80)
plt.scatter(x=[l[0] for l in layer2], y=[l[1] for l in layer2], color="orange", s=80)
plt.scatter(x=[l[0] for l in layer3], y=[l[1] for l in layer3], color="blue", s=80)

ax = plt.gca()
for i in range(len(layer1)):
    ax.annotate(layer1[i][2], (layer1[i][0]+1e3, layer1[i][1]+.1))
for i in range(1,len(layer2)):
    ax.annotate(layer2[i][2], (layer2[i][0]+1e3, layer2[i][1]+.1))
ax.annotate(layer2[0][2], (layer2[0][0]-3e5, layer2[0][1]+.1))
for i in range(len(layer3)):
    ax.annotate(layer3[i][2], (layer3[i][0]+1e3, layer3[i][1]+.1))

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