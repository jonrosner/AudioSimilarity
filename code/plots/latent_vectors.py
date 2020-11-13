import json
from sklearn.manifold import TSNE
import numpy as np
import matplotlib
import seaborn as sns
from collections import defaultdict

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

artist_map = {
    "Aha": "grey",
    "Eminem": "green",
    "Nirvana": "red",
    "Billy Talent": "black",
    "Rammstein": "orange",
    "Ska-P": "brown",
    "U2": "white",
    "Daft Punk": "blue",
    "Chick Corea": "orange",
    "Kings Of Leon": "yellow"
}


vectors = []
artists = []

with open('song_vectors.json') as jsonfile:
    result = json.load(jsonfile)
    for artist, values in result.items():
        if artist in ["Eminem", "Nirvana", "Billy Talent", "Ska-P", "Daft Punk" ,"Chick Corea", "Kings Of Leon"]:
            for value in values:
                vector = value["latent"]
                artists.append(artist)
                vectors.append(vector)

print(np.array(vectors).shape)

X_embedded = TSNE(n_components=2).fit_transform(np.array(vectors))

for artist, p in zip(artists, X_embedded.tolist()):
    plt.scatter(p[0], p[1], color=artist_map[artist], s=30)

ax = plt.gca()
ax.set_xticks([])
ax.set_yticks([])
ax.set(xlabel="t-SNE axis 1", ylabel="t-SNE axis 2")
ax.grid(True)


# plt.show()
plt.savefig("latent_vectors.pgf")