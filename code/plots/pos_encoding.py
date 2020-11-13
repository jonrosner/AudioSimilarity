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

i = np.arange(512)[np.newaxis, :]
position = np.arange(300)[:, np.newaxis]

angle_rads = position / np.power(10_000, (2 * i / 512))
angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

plt.imshow(angle_rads)


ax = plt.gca()
ax.set(xlabel="FFT bin", ylabel="Timestep")

fig = plt.gcf()
fig.set_size_inches(w=4, h=3)
plt.tight_layout()

#plt.show()
plt.savefig("pos_encoding.pgf")