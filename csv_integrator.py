import os.path
from csv_processor import CSVProcessor
import pandas as pd
from translator import Translator


class CSVIntegrator:
    def __init__(self, processed_path, dir_hash_path, scene_list):
        self.processed_path = processed_path
        self.dir_hash_path = dir_hash_path
        self.scene_list = scene_list

        self.complete_row = os.path.join(self.dir_hash_path, "complete_row.csv")
        self.ag_row = os.path.join(self.dir_hash_path, "ag_row.csv")
        self.ml_row = os.path.join(self.dir_hash_path, "ml_row.csv")

        self.complete_col = os.path.join(self.dir_hash_path, "complete_col.csv")
        self.ag_col = os.path.join(self.dir_hash_path, "ag_col.csv")
        self.ml_col = os.path.join(self.dir_hash_path, "ml_col.csv")

        self.complete_mean = os.path.join(self.dir_hash_path, "complete_mean.csv")
        self.ag_mean = os.path.join(self.dir_hash_path, "ag_mean.csv")
        self.ml_mean = os.path.join(self.dir_hash_path, "ml_mean.csv")

    def integrate_row(self):
        all_complete = None
        all_ag = None
        for index, scene in enumerate(self.scene_list):
            scene_home = os.path.join(self.processed_path, scene)
            scene_csvs_home = os.path.join(scene_home, "csvs")

            scene_complete = os.path.join(scene_csvs_home, "complete.csv")
            complete = pd.read_csv(scene_complete)
            complete.insert(len(complete.columns), "scene", pd.Series([index] * len(complete)))
            if all_complete is None:
                all_complete = complete
            else:
                all_complete = pd.concat([all_complete, complete])

            scene_ag = os.path.join(scene_csvs_home, "ag.csv")
            ag = pd.read_csv(scene_ag)
            ag.insert(len(ag.columns), "scene", pd.Series([index] * len(ag)))
            if all_ag is None:
                all_ag = ag
            else:
                all_ag = pd.concat([all_ag, ag])

        all_complete.to_csv(self.complete_row, index=False)
        all_ag.to_csv(self.ag_row, index=False)
        CSVProcessor.make_ml_ready(self.ag_row, self.ml_row)
        return self.complete_row, self.ag_row, self.ml_row

    def integrate_col(self):
        all_complete = None
        all_ag = None
        original_band_columns = Translator.get_bands()
        for index, scene in enumerate(self.scene_list):
            updated_band_columns = [f"{band}_{index}" for band in Translator.get_bands()]
            update_map = {original_band_columns[i]: updated_band_columns[i] for i in range(len(original_band_columns))}

            scene_home = os.path.join(self.processed_path, scene)
            scene_csvs_home = os.path.join(scene_home, "csvs")

            scene_complete = os.path.join(scene_csvs_home, "complete.csv")
            complete = pd.read_csv(scene_complete)
            complete.insert(len(complete.columns), "scene", pd.Series([index] * len(complete)))
            complete.rename(columns=update_map, inplace=True)
            if all_complete is None:
                all_complete = complete
            else:
                all_complete.reset_index(inplace=True, drop=True)
                complete.reset_index(inplace=True, drop=True)
                complete = complete[updated_band_columns]
                all_complete = pd.concat([all_complete, complete], axis=1)

            scene_ag = os.path.join(scene_csvs_home, "ag.csv")
            ag = pd.read_csv(scene_ag)
            ag.insert(len(ag.columns), "scene", pd.Series([index] * len(ag)))
            ag.rename(columns=update_map, inplace=True)
            if all_ag is None:
                all_ag = ag
            else:
                all_ag.reset_index(inplace=True, drop=True)
                ag.reset_index(inplace=True, drop=True)
                ag = ag[updated_band_columns]
                all_ag = pd.concat([all_ag, ag], axis=1)

        all_complete.to_csv(self.complete_col, index=False)
        all_ag.to_csv(self.ag_col, index=False)
        CSVProcessor.make_ml_ready(self.ag_col, self.ml_col)
        return self.complete_row, self.ag_col, self.ml_col

    def integrate_mean(self):
        all_complete = None
        all_ag = None
        for index, scene in enumerate(self.scene_list):
            scene_home = os.path.join(self.processed_path, scene)
            scene_csvs_home = os.path.join(scene_home, "csvs")

            scene_complete = os.path.join(scene_csvs_home, "complete.csv")
            complete = pd.read_csv(scene_complete)
            complete.insert(len(complete.columns), "scene", pd.Series([index] * len(complete)))
            if all_complete is None:
                all_complete = complete
            else:
                all_complete = pd.concat([all_complete, complete])

            scene_ag = os.path.join(scene_csvs_home, "ag.csv")
            ag = pd.read_csv(scene_ag)
            ag.insert(len(ag.columns), "scene", pd.Series([index] * len(ag)))
            if all_ag is None:
                all_ag = ag
            else:
                all_ag = pd.concat([all_ag, ag])

        all_complete.to_csv(self.complete_mean, index=False)

        all_ag.drop(inplace=True, columns=["scene"], axis=1)
        spatial_columns = CSVProcessor.get_spatial_columns(all_ag)
        columns_to_agg = all_ag.columns.drop(spatial_columns)
        all_ag = all_ag.groupby(spatial_columns)[columns_to_agg].mean().reset_index()
        all_ag.to_csv(self.ag_mean, index=False)

        CSVProcessor.make_ml_ready(self.ag_mean, self.ml_mean)

        return self.complete_mean, self.ag_mean, self.ml_mean
