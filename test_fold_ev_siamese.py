from fold_evaluator import FoldEvaluator

if __name__ == "__main__":
    train_scenes = [
        "S2A_MSIL2A_20220207T002711_N0400_R016_T54HWE_20220207T023040",
        "S2B_MSIL2A_20220413T002709_N0400_R016_T54HXE_20220413T021511",
        "S2B_MSIL2A_20220423T002659_N0400_R016_T54HXE_20220423T021724",
        "S2B_MSIL2A_20220503T002659_N0400_R016_T54HXE_20220503T023159"
    ]

    test_scene = "S2B_MSIL2A_20220523T002709_N0400_R016_T54HWE_20220523T021750"

    configs = []

    for lim in range(4):
        configs.append({"input": "bands", "train_scenes": train_scenes[0:lim+1], "test_scene":test_scene})
        #configs.append({"input": "all_ex_som", "train_scenes": train_scenes[0:lim+1], "test_scene":test_scene})
    c = FoldEvaluator(configs=configs, prefix="band_all", folds=10, algorithms=["mlr","rf","svr","ann"])
    c.process()