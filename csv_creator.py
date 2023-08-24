import os.path

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np


class CSVCreator:
    def __init__(self, df, export_dir):
        self.df = df
        self.base_dir = export_dir
        self.complete = os.path.join(self.base_dir,"complete.csv")
        self.ag = os.path.join(self.base_dir, "ag.csv")
        self.ml = os.path.join(self.base_dir,"ml.csv")
        self.geo_columns = ["lon", "lat", "when"]
        self.spatial_columns = ["scene", "row", "column"]

    def make_ml_ready(self):
        df = pd.read_csv(self.ag)
        df = self.make_ml_ready_df(df)
        df.to_csv(self.ml, index=False)

    def make_ml_ready_df(self, df):
        df.drop(inplace=True, columns=self.spatial_columns, axis=1)
        for col in self.geo_columns:
            if col in df.columns:
                df.drop(inplace=True, columns=[col], axis=1)
        data = df.to_numpy()
        for i in range(data.shape[1]):
            scaler = MinMaxScaler()
            x_scaled = scaler.fit_transform(data[:, i].reshape(-1, 1))
            data[:, i] = np.squeeze(x_scaled)
        df = pd.DataFrame(data=data, columns=df.columns)
        return df

    def aggregate(self):
        df = pd.read_csv(self.complete)
        df.drop(columns=self.geo_columns, axis=1, inplace=True)
        columns_to_agg = df.columns.drop(self.spatial_columns)
        df = df.groupby(self.spatial_columns)[columns_to_agg].mean().reset_index()
        df.to_csv(self.ag, index=False)

    def create(self):
        self.df.to_csv(self.complete, index=False)
        self.aggregate()
        self.make_ml_ready()
        return \
            os.path.join(self.base_dir,"complete.csv"), \
            os.path.join(self.base_dir, "ag.csv"), \
            os.path.join(self.base_dir,"ml.csv")

