from clipper import Clipper
from reconstruct import Reconstructor
import pandas as pd
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import os
from rasterio.windows import Window

csv = r"E:\src\grid_siamese\data\processed\8e09234d1e1696d5c65e715b39d56b55\ag.csv"
df = pd.read_csv(csv)
if "scene" in df.columns:
    df = df[df["scene"] == 0]
file_name = f"111.jp2"

rgb = None
with rasterio.open(file_name) as src:
    r = src.read(1)
    g = src.read(2)
    b = src.read(3)
    rgb = np.stack((r, g, b), axis=-1)

x = np.zeros((rgb.shape[0], rgb.shape[1]))
for i in df.index:
    row = df.loc[i, "row"]
    col = df.loc[i, "column"]
    pix = df.loc[i, "som"]
    row = int(row)
    col = int(col)
    x[row, col] = pix

fig, ax = plt.subplots()
alpha_array = np.ones_like(x)
alpha_array[x == 0] = 0
# ax.imshow(rgb)
cax = ax.imshow(x, alpha=alpha_array)
cbar = plt.colorbar(cax, ax=ax)
cbar.set_label('SOM')
plt.axis('off')
plt.show()
