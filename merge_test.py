import pandas as pd


df = pd.read_csv("data/processed/S2A_MSIL2A_20220207T002711_N0400_R016_T54HWE_20220207T023040/csvs/complete.csv")
print(df.max())