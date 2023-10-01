from clipper import Clipper
from reconstruct import Reconstructor
import pandas as pd
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import os
from rasterio.windows import Window

source = (r"E:\tim\vectis\S2B_MSIL2A_20220423T002659_N0400_R016_T54HXE_20220423T021724\S2B_MSIL2A_20220423T002659_N0400_R016_T54HXE_20220423T021724.SAFE\GRANULE\L2A_T54HXE_A026783_20220423T003625\IMG_DATA\R10m\T54HXE_20220423T002659_TCI_10m.jp2")

dest = r"14.jp2"
source_csv_path = "data/shorter.csv"
clipper = Clipper(source, dest, source_csv_path)
clipper.clip()
