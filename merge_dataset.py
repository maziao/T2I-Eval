import os
import json
import copy
import argparse
import random
import numpy as np
from src.utils.extract_scores import extract_score_list_from_str


def sort_summary_samples_by_score(sample_list: list):
    score_bin = {f"{i}-{i+1}": [] for i in range(10)}
    score_bin['10'] = []
    score_bin['N/A'] = []
    sorted_samples = {
        "appearance": copy.deepcopy(score_bin),
        "intrinsic": copy.deepcopy(score_bin),
        "relationship": copy.deepcopy(score_bin),
        "overall": copy.deepcopy(score_bin)
    }
    for sample in sample_list:
        category = sample['id'].split('-')[1]
        try:
            if 'response' in sample:
                score = extract_score_list_from_str(sample['response'], force_four_scores=False)[-1]
            elif 'conversations' in sample:
                score = extract_score_list_from_str(sample['conversations'][-1]['value'], force_four_scores=False)[-1]
        except Exception:
            score = None
        if score is None or score == 'N/A':
            bin_name = 'N/A'
        elif score >= 10.0:
            bin_name = '10'
        elif score >= 0.0 and score < 10.0:
            bin_name = f'{int(score)}-{int(score) + 1}'
        else:
            print(f"Invalid score: {score}")
            bin_name = 'N/A'
        sorted_samples[category][bin_name].append(sample)
    return sorted_samples


def sort_samples_by_score(stage_2_sample_list: list, stage_1_sample_list: list = []):
    score_bin = {f"{i}-{i+1}": [] for i in range(10)}
    score_bin['10'] = []
    score_bin['N/A'] = []
    sorted_samples = score_bin
    for sample in stage_2_sample_list:
        stage_1_id = sample['id'].replace('-stage_2', '-stage_1')
        try:
            if 'response' in sample:
                score = extract_score_list_from_str(sample['response'], force_four_scores=False)[-1]
            elif 'conversations' in sample:
                score = extract_score_list_from_str(sample['conversations'][-1]['value'], force_four_scores=False)[-1]
        except Exception:
            score = None
        if score is None or score == 'N/A':
            bin_name = 'N/A'
        elif score >= 10.0:
            bin_name = '10'
        elif score >= 0.0 and score < 10.0:
            bin_name = f'{int(score)}-{int(score) + 1}'
        else:
            print(f"Invalid score: {score}")
            bin_name = 'N/A'
        sorted_samples[bin_name].append(sample)
        if len(stage_1_sample_list) != 0:
            for stage_1_sample in stage_1_sample_list:
                if stage_1_sample['id'] == stage_1_id:
                    sorted_samples[bin_name].append(stage_1_sample)
                    break
    return sorted_samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--index-file', type=str, required=True)
    parser.add_argument('--output-file', type=str, required=True)
    parser.add_argument('--rebalance', action="store_true")
    parser.add_argument('--seed', type=int, default=3407)
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    data_files = {}
    with open(args.index_file, 'r+', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            splits = line.strip().split()
            assert len(splits) == 2
            data_files[os.path.join(args.data_dir, splits[0])] = int(splits[1])
    
    data = []
    for data_file, repeat_n in data_files.items():
        basename = os.path.basename(data_file)
        with open(data_file, 'r+', encoding='utf-8') as f:
            data_split = json.load(f)
            if basename in [
                "summarize-multi-stage_2.json",
                "summarize.json"
            ] and args.rebalance:
                scoring_data = []
                
                sorted_data_split = sort_summary_samples_by_score(data_split)
                print(f"[!] loaded {len(data_split)} scoring samples from file {data_file}, statistics:")
                
                for category in sorted_data_split:
                    total_sample_expanded = 0
                    total_sample = sum([len(sorted_data_split[category][bin_name]) for bin_name in sorted_data_split[category]])
                    median_sample = np.percentile([len(sorted_data_split[category][bin_name]) for bin_name in sorted_data_split[category]], 75)
                    print(f"- category: {category}")
                    for bin_name in sorted_data_split[category]:
                        if bin_name == 'N/A':
                            print(f"  - {bin_name}: {len(sorted_data_split[category][bin_name])}, desert.")
                            continue
                        if args.rebalance:
                            bin_size = len(sorted_data_split[category][bin_name])
                            _repeat_n = median_sample / bin_size
                            new_bin_size = int(bin_size * _repeat_n)
                        
                            sample_size = new_bin_size - int(_repeat_n) * bin_size
                            total_sample_expanded += new_bin_size
                            print(f"  - {bin_name}: {bin_size}, repeat {round(_repeat_n, 2)} times, total {new_bin_size} samples.")
                            
                            for i in range(int(_repeat_n)):
                                scoring_data.extend(copy.deepcopy(sorted_data_split[category][bin_name]))
                            scoring_data.extend(random.sample(copy.deepcopy(sorted_data_split[category][bin_name]), sample_size))
                        else:
                            scoring_data.extend(sorted_data_split[category][bin_name])
                            total_sample_expanded += len(sorted_data_split[category][bin_name])
                    print(f"  - category summary: repeat {round(total_sample_expanded / total_sample, 2)} times for score rebalancing (summarize stage).")
                for i in range(repeat_n):
                    data.extend(copy.deepcopy(scoring_data))
                print(f"[!] scoring samples repeat {repeat_n} times ({round(repeat_n * len(scoring_data) / len(data_split), 2)} times in total), total {len(scoring_data) * repeat_n} samples.")
            elif basename in [
                "appearance-multi-stage_1.json",
                "intrinsic-multi-stage_1.json",
                "relationship-multi-stage_1.json",
            ] and args.rebalance:
                continue
            elif basename in [
                "appearance-multi-stage_2.json",
                "intrinsic-multi-stage_2.json",
                "relationship-multi-stage_2.json",
            ] and args.rebalance:
                data_file_stage_1 = data_file.replace('-stage_2', '-stage_1')
                with open(data_file_stage_1, 'r+', encoding='utf-8') as f:
                    data_split_stage_1 = json.load(f)
                    
                answer_and_eval_data = []
                
                sorted_data_split = sort_samples_by_score(data_split, data_split_stage_1)
                print(f"[!] loaded {len(data_split)} scoring samples from file {data_file}, statistics:")

                total_sample_expanded = 0
                total_sample = sum([len(sorted_data_split[bin_name]) for bin_name in sorted_data_split])
                median_sample = np.percentile([len(sorted_data_split[bin_name]) for bin_name in sorted_data_split], 75)
                for bin_name in sorted_data_split:
                    if bin_name == 'N/A':
                        print(f"  - {bin_name}: {len(sorted_data_split[bin_name])}, desert.")
                        continue
                    bin_size = len(sorted_data_split[bin_name])
                    if bin_size == 0:
                        continue
                    _repeat_n = median_sample / bin_size
                    new_bin_size = int(bin_size * _repeat_n)
                
                    sample_size = new_bin_size - int(_repeat_n) * bin_size
                    total_sample_expanded += new_bin_size
                    print(f"  - {bin_name}: {bin_size}, repeat {round(_repeat_n, 2)} times, total {new_bin_size} samples.")
                    
                    for i in range(int(_repeat_n)):
                        answer_and_eval_data.extend(copy.deepcopy(sorted_data_split[bin_name]))
                    answer_and_eval_data.extend(random.sample(copy.deepcopy(sorted_data_split[bin_name]), sample_size))
                print(f"  - category summary: repeat {round(total_sample_expanded / total_sample, 2)} times for score rebalancing (answer+eval stage).")
                
                for i in range(repeat_n):
                    data.extend(copy.deepcopy(answer_and_eval_data))
                print(f"[!] scoring samples repeat {repeat_n} times ({round(repeat_n * len(answer_and_eval_data) / (2 * len(data_split)), 2)} times in total), total {len(answer_and_eval_data) * repeat_n} samples.")
            elif basename in [
                "appearance.json",
                "intrinsic.json",
                "relationship.json",
            ] and args.rebalance:
                answer_and_eval_data = []
                
                sorted_data_split = sort_samples_by_score(data_split)
                print(f"[!] loaded {len(data_split)} scoring samples from file {data_file}, statistics:")

                total_sample_expanded = 0
                total_sample = sum([len(sorted_data_split[bin_name]) for bin_name in sorted_data_split])
                median_sample = np.percentile([len(sorted_data_split[bin_name]) for bin_name in sorted_data_split], 75)
                for bin_name in sorted_data_split:
                    if bin_name == 'N/A':
                        print(f"  - {bin_name}: {len(sorted_data_split[bin_name])}, desert.")
                        continue
                    bin_size = len(sorted_data_split[bin_name])
                    if bin_size == 0:
                        continue
                    _repeat_n = median_sample / bin_size
                    new_bin_size = int(bin_size * _repeat_n)
                
                    sample_size = new_bin_size - int(_repeat_n) * bin_size
                    total_sample_expanded += new_bin_size
                    print(f"  - {bin_name}: {bin_size}, repeat {round(_repeat_n, 2)} times, total {new_bin_size} samples.")
                    
                    for i in range(int(_repeat_n)):
                        answer_and_eval_data.extend(copy.deepcopy(sorted_data_split[bin_name]))
                    answer_and_eval_data.extend(random.sample(copy.deepcopy(sorted_data_split[bin_name]), sample_size))
                print(f"  - category summary: repeat {round(total_sample_expanded / total_sample, 2)} times for score rebalancing (answer+eval stage).")
                
                for i in range(repeat_n):
                    data.extend(copy.deepcopy(answer_and_eval_data))
                print(f"[!] scoring samples repeat {repeat_n} times ({round(repeat_n * len(answer_and_eval_data) / (2 * len(data_split)), 2)} times in total), total {len(answer_and_eval_data) * repeat_n} samples.")
            else:
                for i in range(repeat_n):
                    data.extend(copy.deepcopy(data_split))
                print(f"[!] loaded {len(data_split)} samples from file {data_file}, repeat {repeat_n} times, total {len(data_split) * repeat_n} samples.")

    random.shuffle(data)
    print(f"[!] loaded {len(data)} samples in total.")
    
    for i in range(len(data)):
        data[i]['id'] = str(i)
        
    with open(args.output_file, 'w+', encoding='utf-8') as f:
        json.dump(obj=data, fp=f, ensure_ascii=False, indent=4)
    print(f"[!] merging completed. [!]")
    