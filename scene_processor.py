import os
from base_path import BASE_PATH
import re
from clipper import Clipper


class SceneProcessor:
    def __init__(self, scene_list, processed_path, source_csv_path):
        self.scene_list = scene_list
        self.processed_path = processed_path
        self.source_csv_path = source_csv_path

    def create_clips(self):
        for index, scene in enumerate(self.scene_list):
            dest_clipped_scene_folder_path = os.path.join(self.processed_path, scene)
            if os.path.exists(dest_clipped_scene_folder_path):
                print(f"Processed scene dir exists {index + 1}: {scene}. Skipping")
                continue
            os.mkdir(dest_clipped_scene_folder_path)
            base = self.get_scene_source(scene)
            self.clip_bands(base, dest_clipped_scene_folder_path)
            print(f"Done clipping scene {index+1}: {scene}")

    def clip_bands(self, base, dest_clipped_scene_folder_path):
        done = []
        folders = os.listdir(base)
        folders = sorted(folders, key=lambda x: int(re.findall(r'\d+', x)[0]), reverse=True)
        for resolution in folders:
            resolution_path = os.path.join(base, resolution)
            for file_name in os.listdir(resolution_path):
                if not file_name.endswith(".jp2"):
                    continue
                parts = file_name.split("_")
                band = parts[2]
                if band in done:
                    continue
                done.append(band)
                source_band_path = os.path.join(resolution_path, file_name)
                dest_band_path = os.path.join(dest_clipped_scene_folder_path, f"{band}.jp2")
                clipper = Clipper(source_band_path, dest_band_path, self.source_csv_path)
                clipper.clip()
        return done

    def get_scene_source(self, scene):
        scene_path = os.path.join(BASE_PATH, scene)
        return SceneProcessor.get_img_data_path(scene_path)

    @staticmethod
    def get_img_data_path(scene_path):
        safe = os.listdir(scene_path)[0]
        safe_path = os.path.join(scene_path, safe)
        granule_path = os.path.join(safe_path,"GRANULE")
        sub = os.listdir(granule_path)[0]
        sub_path = os.path.join(granule_path, sub)
        img_path = os.path.join(os.path.join(sub_path,"IMG_DATA"))
        return img_path

    @staticmethod
    def get_all_scenes():
        scene_list = os.listdir(BASE_PATH)
        scene_list = [scene for scene in scene_list if scene.startswith("S2")
                           and os.path.isdir(os.path.join(BASE_PATH, scene))]
        return scene_list
