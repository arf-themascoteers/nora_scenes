import os
import pandas as pd
import hashlib
from environment import TEST
from csv_integrator import CSVIntegrator
from scene_processor import SceneProcessor
from scenes_to_csv import SceneToCSVs


class S2Extractor:
    def __init__(self, scenes=0):
        self.TEST = TEST
        self.FILTERED = True
        self.source_csv = "vectis.csv"
        if self.TEST:
            self.source_csv = "vectis_min.csv"
        self.source_csv_path = os.path.join("data", self.source_csv)
        self.datasets_list_file = "datasets.csv"

        if self.FILTERED:
            short_csv = "shorter.csv"
            short_csv_path = os.path.join("data", short_csv)
            S2Extractor.shorten(self.source_csv_path, short_csv_path)
            self.source_csv_path = short_csv_path

        processed_dir = "processed"
        self.processed_path = os.path.join("data", processed_dir)
        self.datasets_list_file_path = os.path.join(self.processed_path, "datasets.csv")

        if not os.path.exists(self.processed_path):
            os.mkdir(self.processed_path)

        if type(scenes) == list:
            self.scene_list = scenes
        else:
            self.scene_list = SceneProcessor.get_all_scenes()
            if scenes == 0:
                scenes = len(self.scene_list)
            self.scene_list = self.scene_list[0:scenes]
        self.scene_list = sorted(self.scene_list)
        self.scenes_str = S2Extractor.create_scenes_string(self.scene_list)
        self.dir_str_original = self.scenes_str
        self.dir_hash =  hashlib.md5(self.dir_str_original.encode('UTF-8')).hexdigest()
        self.dir_hash_path = os.path.join(self.processed_path, self.dir_hash)
        self.clip_path = os.path.join(self.dir_hash_path, "clipped")

    @staticmethod
    def shorten(orig, short):
        df = pd.read_csv(orig)
        df = df[df["som"] > 1.72]
        df = df[df["som"] < 3.29]
        df.to_csv(short, index=False)

    def write_dataset_list_file(self, dirname, scenes):
        row = self.read_dataset_list_file(dirname, scenes)
        if row is not None:
            return
        if os.path.exists(self.datasets_list_file_path):
            df = pd.read_csv(self.datasets_list_file_path)
            df.loc[len(df)] = [dirname, scenes]
            df.columns = ["dirname", "scenes"]
        else:
            df = pd.DataFrame(data=[[dirname,scenes]], columns=["dirname", "scenes"])
        df.to_csv(self.datasets_list_file_path, index=False)

    def read_dataset_list_file(self, dirname, scenes):
        if not os.path.exists(self.datasets_list_file_path):
            return None

        df = pd.read_csv(self.datasets_list_file_path)
        df = df[((df['dirname'] == dirname) & (df['scenes'] == scenes))]
        if len(df) == 0:
            return None
        return df.iloc[0]

    @staticmethod
    def create_scenes_string(scenes):
        return ":".join(scenes)

    def process(self):
        if os.path.exists(self.dir_hash_path):
            print(f"Dir exists for {self.dir_str_original} - ({self.dir_hash_path}). Skipping.")
            complete = os.path.join(self.dir_hash_path, "complete.csv")
            ag = os.path.join(self.dir_hash_path, "ag.csv")
            ml = os.path.join(self.dir_hash_path, "ml.csv")
            return complete, ag, ml, self.scene_list

        os.mkdir(self.dir_hash_path)
        os.mkdir(self.clip_path)
        scene_processor = SceneProcessor(self.scene_list, self.processed_path, self.source_csv_path)
        scene_processor.create_clips()
        cd = SceneToCSVs(self.scene_list, self.processed_path, self.source_csv_path)
        cd.create_csvs()
        csv = CSVIntegrator(self.processed_path, self.dir_hash_path, self.scene_list)
        complete, ag, ml = csv.integrate()
        self.write_dataset_list_file(self.dir_hash, self.scenes_str)
        return complete, ag, ml, self.scene_list


if __name__ == "__main__":
    s2 = S2Extractor()
    s2.process()



