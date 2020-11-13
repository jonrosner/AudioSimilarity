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
 "legend.facecolor": "white",
 "legend.framealpha": 1,
 "legend.loc": "upper right"})

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

import matplotlib.pyplot as plt
import numpy as np

# cs = {
#     "red": (0.91, 0.604, 0.604),
#     "yellow": (0.95686275, 0.76078431, 0.05098039),
#     "blue": (0.639, 0.761, 0.973),
#     "green": (0.655, 0.863, 0.659),
#     "orange": (0.941, 0.741, 0.533),
#     "purple": (0.773, 0.682, 0.784)
# }

cs = {
    "red": (0.85882353, 0.19607843, 0.21176471),
    "yellow": (0.95686275, 0.76078431, 0.05098039),
    "blue": (0.28235294, 0.52156863, 0.92941176),
    "green": (0.23529412, 0.72941176, 0.32941176),
    "orange": (0.90196078, 0.47843137, 0.08627451),
    "purple": (0.56862745, 0.35686275, 0.56862745)
}

x,y = np.split(np.random.normal(0, 1, (1000,2)), indices_or_sections=2, axis=1)
x[x > 2] = 20
x[x < -2] = 20
y[y > 2] = 20
y[y < -2] = 20
plt.scatter(x,y, s=5, color=cs["blue"])
x_green = [-2,-2,-1.5,-1.75,-1.55, -1.35, -1.7]
y_green = [-1,-1.25,-1,-1.5,-1.1, -1, -0.5]
plt.scatter(x_green, y_green, color=cs["green"], s=80)

x_red = [0.7,0.7,1.2,0.95,0.2,0.8,1.1]
y_red = [-0,-0.3,-0,-0.5,-0.4,0.2,-0.4]
plt.scatter(x_red, y_red, color=cs["red"], s=80)

x_orange = [-0.4,-0.3,0,0.1,-0.5,-0.7,-0.12]
y_orange = [1,1.25,1.1,0.5,1,1,0.5]
plt.scatter(x_orange, y_orange, color=cs["orange"], s=80)

plt.gca().set_xlim(-2.2, 2.2)
plt.gca().set_ylim(-2.2, 2.2)
plt.gca().axes.get_yaxis().set_visible(False)
plt.gca().axes.get_xaxis().set_visible(False)
plt.legend(["Data", "Airplanes", "Dogs", "Cats"])
plt.axis('off')


fig = plt.gcf()
plt.gca().axes.get_yaxis().set_visible(False)
plt.gca().axes.get_xaxis().set_visible(False)
plt.gca().set_axis_off()
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
plt.margins(0,0)
fig.set_size_inches(w=4, h=4)
plt.tight_layout()

#plt.show()
plt.savefig('clusters.pgf')


