from fold_evaluator import FoldEvaluator

if __name__ == "__main__":
    inputs = ["bands", "props_ex_som", "all_ex_som"]
    scenes = [
        "S2A_MSIL2A_20220207T002711_N0400_R016_T54HWE_20220207T023040"
    ]

    configs = []
    for i in inputs:
        configs.append({"input": i, "scenes": scenes})
    c = FoldEvaluator(configs=configs, prefix="t1", folds=10, algorithms=["mlr","svr","ann"])
    c.process()