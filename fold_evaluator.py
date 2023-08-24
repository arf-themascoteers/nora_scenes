from s2_extractor import S2Extractor
from fold_reporter import FoldReporter
from config_creator import ConfigCreator
from algorithm_runner import AlgorithmRunner
from fold_ds_manager import FoldDSManager


class FoldEvaluator:
    def __init__(self, prefix="", verbose=False, repeat=1, folds=10, algorithms=None, configs=None):
        self.repeat = repeat
        self.folds = folds
        self.verbose = verbose
        self.algorithms = algorithms

        if self.algorithms is None:
            self.algorithms = ["mlr", "rf", "svr", "ann", "siamese"]

        self.config_list = []
        self.train_csvs = []
        self.test_csvs = []
        self.scenes = []
        scenes_count = []
        scenes_string = []

        for config in configs:
            config_object = ConfigCreator.create_config_object(config)
            self.config_list.append(config_object)
            s2 = S2Extractor(config_object["train_scenes"])
            _, ml_train, _, scenes = s2.process()
            s2 = S2Extractor([config_object["test_scene"]])
            _, ml_test, _, _ = s2.process()
            self.train_csvs.append(ml_train)
            self.test_csvs.append(ml_test)
            self.scenes.append(scenes)
            scenes_count.append(len(scenes))
            scenes_string.append(scenes)

        self.reporter = FoldReporter(prefix, self.config_list, scenes_count, scenes_string,
                                 self.algorithms, self.repeat, self.folds)

    def process(self):
        for repeat_number in range(self.repeat):
            self.process_repeat(repeat_number)

    def process_repeat(self, repeat_number):
        for index_algorithm, algorithm in enumerate(self.algorithms):
            self.process_algorithm(repeat_number, index_algorithm)

    def process_algorithm(self, repeat_number, index_algorithm):
        for index_config in range(len(self.config_list)):
            config = self.config_list[index_config]
            print("Start", f"{repeat_number}:{self.algorithms[index_algorithm]} - {config}")
            self.process_config(repeat_number, index_algorithm, index_config)

    def process_config(self, repeat_number, index_algorithm, index_config):
        algorithm = self.algorithms[index_algorithm]
        config = self.config_list[index_config]
        train_ds = FoldDSManager(self.train_csvs[index_config], folds=self.folds, x=config["input"], y=config["output"])
        test_ds = FoldDSManager(self.test_csvs[index_config], folds=self.folds, x=config["input"], y=config["output"])
        for fold_number, (test_x, test_y) in enumerate(test_ds.get_k_folds()):
            train_x, train_y, validation_x, validation_y = train_ds.get_train_validation()
            r2, rmse = self.reporter.get_details(index_algorithm, repeat_number, fold_number, index_config)
            if r2 != 0:
                print(f"{repeat_number}-{fold_number} done already")
                continue
            else:
                r2, rmse = AlgorithmRunner.calculate_score(train_x, train_y,
                                                           test_x, test_y,
                                                           validation_x, validation_y,
                                                           algorithm,
                                                           train_ds.band_index_start, train_ds.band_count
                                                           )
            if self.verbose:
                print(f"{r2} - {rmse}")
                print(f"R2 - RMSE")
            self.reporter.set_details(index_algorithm, repeat_number, fold_number, index_config, r2, rmse)
            self.reporter.write_details()
            self.reporter.update_summary()

