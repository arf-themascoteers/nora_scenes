from fold_evaluator import FoldEvaluator

if __name__ == "__main__":
    inputs = ["all_ex_som"]
    scenes = [
        "S2A_MSIL2A_20220207T002711_N0400_R016_T54HWE_20220207T023040",
        "S2B_MSIL2A_20220413T002709_N0400_R016_T54HXE_20220413T021511",
        "S2B_MSIL2A_20220423T002659_N0400_R016_T54HXE_20220423T021724",
        "S2B_MSIL2A_20220503T002659_N0400_R016_T54HXE_20220503T023159",
        "S2B_MSIL2A_20220523T002709_N0400_R016_T54HWE_20220523T021750"

    ]

    configs = []

    configs.append({"input": "all_ex_som", "scenes": scenes[0:1]})
    configs.append({"input": "all_ex_som", "scenes": scenes[0:2]})
    configs.append({"input": "all_ex_som", "scenes": scenes[0:3]})
    configs.append({"input": "all_ex_som", "scenes": scenes[0:4]})
    configs.append({"input": "all_ex_som", "scenes": scenes[0:5]})
    c = FoldEvaluator(configs=configs, prefix="5_Scenes_all", folds=2, algorithms=["svr","ann"])
    c.process()