from fold_evaluator import FoldEvaluator

if __name__ == "__main__":
    scenes = [
        "S2A_MSIL2A_20220207T002711_N0400_R016_T54HWE_20220207T023040",
        "S2B_MSIL2A_20220413T002709_N0400_R016_T54HXE_20220413T021511",
        "S2B_MSIL2A_20220423T002659_N0400_R016_T54HXE_20220423T021724",
        "S2B_MSIL2A_20220503T002659_N0400_R016_T54HXE_20220503T023159",
        "S2B_MSIL2A_20220523T002709_N0400_R016_T54HWE_20220523T021750"

    ]

    configs = []

    for inputs in ["bands", "props_ex_som", "all_ex_som",
                   "bands_elevation","bands_moisture","bands_temp",
                   "bands_elevation_moisture","bands_moisture_temp","bands_temp_elevation",
                   ["elevation"],["moisture"],["temp"],
                   ["elevation","moisture"],["moisture","temp"],["temp","elevation"]
                   ]:
        for ag in ["row"]:#, "col", "mean"]:
            for lim in range(1,5):
                configs.append({"input": inputs, "scenes": scenes[0:lim+1], "ag":ag})
    c = FoldEvaluator(configs=configs, prefix="all2", folds=10, algorithms=["mlr", "rf", "svr", "ann"])
    c.process()