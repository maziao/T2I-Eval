import os
import json
import numpy as np
from scipy import stats
from tabulate import tabulate


def fill_na(score_list: list, strategy: str = "mean"):
    for i in range(len(score_list)):
        if not isinstance(score_list[i], int) and not isinstance(score_list[i], float):
            score_list[i] = None
    assert strategy in ["mean", "zero"]
    if strategy == "mean":
        arr = np.array(score_list)
        value = np.mean(np.delete(arr, np.where(arr == None)))
    else:
        value = 0
    for i in range(len(score_list)):
        if score_list[i] is None:
            score_list[i] = value
    return score_list


def get_fine_grained_score_mapper(
    ref_score_list: list, result_score_list: list, key: str, split: str = None
):
    mapper = {}
    for sample in ref_score_list:
        for question in sample[key]:
            mapper[question["id"]] = question
    for sample in result_score_list:
        mapper[sample["id"]]["result_score"] = sample["score"]

    for sample in mapper:
        assert "result_score" in mapper[sample], mapper[sample]

    return {
        "openai_score": [value["score"] for _, value in mapper.items()],
        "result_score": [value["result_score"] for _, value in mapper.items()],
        "manual_score_1": [value["manual_score"][0] for _, value in mapper.items()],
        "manual_score_2": [value["manual_score"][1] for _, value in mapper.items()],
        "manual_score_3": [value["manual_score"][2] for _, value in mapper.items()],
        "manual_score_avg": [value["manual_score"][3] for _, value in mapper.items()],
    }


def get_coarse_grained_score_mapper(ref_score_list: list, result_score_list: list, overall_only: bool = False):
    mapper = {}
    for sample in ref_score_list:
        mapper[sample["id"]] = {"ref": sample["overall"]}
    for sample in result_score_list:
        mapper[str(sample["id"])]["result"] = sample
    
    missing_counter = 0
    missing_indices = []
    for sample in mapper:
        assert "ref" in mapper[sample]
        if "result" not in mapper[sample]:
            missing_counter += 1
            missing_indices.append(sample)
    if missing_counter > 0:
        print(f"[!] number of missing samples in the result: {missing_counter}")
        for index in missing_indices:
            mapper.pop(index)
            
    categories = ["appearance", "intrinsic", "relationship", "overall"]
    if overall_only:
        categories = [categories[-1]]

    return {
        category: {
            "openai_score": [
                value["ref"][category]["score"] for _, value in mapper.items()
            ],
            "result_score": [
                value["result"][f"{category}_score"] for _, value in mapper.items()
            ],
            "manual_score_1": [
                value["ref"][category]["manual_score"][0] for _, value in mapper.items()
            ],
            "manual_score_2": [
                value["ref"][category]["manual_score"][1] for _, value in mapper.items()
            ],
            "manual_score_3": [
                value["ref"][category]["manual_score"][2] for _, value in mapper.items()
            ],
            "manual_score_avg": [
                value["ref"][category]["manual_score"][3] for _, value in mapper.items()
            ],
        }
        for category in categories
    }


def calc_correlation(score_dict: dict, corr_type: str = 'spearman'):
    keys = list(score_dict.keys())
    num_keys = len(keys)
    correlation_matrix = np.zeros((num_keys, num_keys))
    pvalue_matrix = np.zeros((num_keys, num_keys))
    for i, key_1 in enumerate(keys):
        for j, key_2 in enumerate(keys):
            if i > j:
                if corr_type == 'spearman':
                    corr = stats.spearmanr(
                        fill_na(score_dict[key_1]), fill_na(score_dict[key_2])
                    )
                elif corr_type == 'pearson':
                    corr = stats.pearsonr(
                        fill_na(score_dict[key_1]), fill_na(score_dict[key_2])
                    )
                elif corr_type == 'kendall':
                    corr = stats.kendalltau(
                        fill_na(score_dict[key_1]), fill_na(score_dict[key_2])
                    )
                else:
                    raise ValueError(f"[!] invalid correlation type: {corr_type}")
                correlation_matrix[i][j] = corr.correlation
                pvalue_matrix[i][j] = corr.pvalue
    corr_table = [[""] + keys]
    for i, key in enumerate(keys):
        corr_table.append([key] + correlation_matrix[i].tolist())
    corr_table = tabulate(corr_table)

    pvalue_table = [[""] + keys]
    for i, key in enumerate(keys):
        pvalue_table.append([key] + pvalue_matrix[i].tolist())
    pvalue_table = tabulate(pvalue_table)

    return corr_table, pvalue_table


def get_result(mapper: dict, name: str = None, corr_type: str = 'spearman'):
    corr, pvalue = calc_correlation(mapper, corr_type=corr_type)
    print(f"# Result for {name}:")
    print(f"correlation:")
    print(corr)
    print(f"pvalue:")
    print(pvalue)


def calc_correlation_from_result_dir(result_dir: str, ref_score_file: str, ):
    with open(ref_score_file, "r+", encoding="utf-8") as f:
        ref_scores = json.load(f)
    
    result_file = os.path.join(result_dir, "summarize-result-score.jsonl")

    with open(result_file, "r+", encoding="utf-8") as f:
        result_scores = [json.loads(line) for line in f.readlines()]
    
    mapper = get_coarse_grained_score_mapper(ref_scores, result_scores)
    
    print('\n' + '*' * 20 + '\nSpearman Correlation\n' + '*' * 20 + '\n')
    for key, value in mapper.items():
        get_result(value, key, 'spearman')

    print('\n' + '*' * 19 + '\nKendall Correlation\n' + '*' * 19 + '\n')
    for key, value in mapper.items():
        get_result(value, key, 'kendall')
    