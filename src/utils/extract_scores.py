import os
import json
from tqdm import tqdm


def extract_score_list_from_str(string: str, force_four_scores: bool = True):
    score_list = []
    splits = string.split()
    for split in splits:
        if split.strip() == 'N/A' or split.strip() == 'None':
            score_list.append(None)
        else:
            try:
                score = float(split.strip())
                score_list.append(score)
            except ValueError:
                continue
    if len(score_list) != 4 and force_four_scores:
        tqdm.write(f"[!] Error: number of scores is not equal to 4 in `{string}` -> {score_list} [!]")
        score_list = score_list[:4]
        score_list += ['N/A'] * (4 - len(score_list))
    return score_list


def extract_score_from_str(score_str: str):
    assert isinstance(score_str, str)
    splits = score_str.split()
    for split in splits:
        if split == 'N/A':
            return 'N/A'
        else:
            try:
                score = float(split)
                return score
            except ValueError:
                continue
    return 'N/A'


def extract_scores_from_result_dir(result_dir: str):
    result_files = [
        os.path.join(result_dir, file)
        for file in sorted(os.listdir(result_dir))
    ]

    score_mapper = {}
    separate_aspect = False
    for file in result_files:
        basename = os.path.basename(file)
        if basename in [
            "appearance_summary_stage_2-result.jsonl",
            "intrinsic_summary_stage_2-result.jsonl",
            "relationship_summary_stage_2-result.jsonl",
        ]:
            separate_aspect = True
            continue
        
        score_dict = {}
        if basename in [
            "appearance_answer_stage_2-result.jsonl",
            "intrinsic_eval_stage_2-result.jsonl",
            "relationship_eval_stage_2-result.jsonl",
            "appearance_answer-result.jsonl",
            "intrinsic_answer-result.jsonl",
            "relationship_answer-result.jsonl",
        ]:
            with open(file, "r+", encoding="utf-8") as f:
                result_list = [json.loads(line) for line in f.readlines()]

            for result in result_list:
                score = {"id": result['id'], "score": extract_score_from_str(result['response'])}
                score_dict[result['id']] = score
        
        elif basename == "summarize_stage_2-result.jsonl":
            with open(file, "r+", encoding="utf-8") as f:
                result_list = [json.loads(line) for line in f.readlines()]
            
            if separate_aspect:
                with open(os.path.join(result_dir, "appearance_summary_stage_2-result.jsonl"), 'r+', encoding='utf-8') as f:
                    appearance_result = [json.loads(line) for line in f.readlines()]
                with open(os.path.join(result_dir, "intrinsic_summary_stage_2-result.jsonl"), 'r+', encoding='utf-8') as f:
                    intrinsic_result = [json.loads(line) for line in f.readlines()]
                with open(os.path.join(result_dir, "relationship_summary_stage_2-result.jsonl"), 'r+', encoding='utf-8') as f:
                    relationship_result = [json.loads(line) for line in f.readlines()]
                for result in result_list:
                    score = {
                        "id": result['id'],
                        "appearance_score": None,
                        "intrinsic_score": None,
                        "relationship_score": None,
                        "overall_score": result['score']
                    }
                    score_dict[result['id']] = score
                for result in appearance_result:
                    score_dict[result['id']]["appearance_score"] = result['score']
                for result in intrinsic_result:
                    score_dict[result['id']]["intrinsic_score"] = result['score']
                for result in relationship_result:
                    score_dict[result['id']]["relationship_score"] = result['score']
            else:
                for result in result_list:
                    scores = result['scores']
                    # scores = extract_score_list_from_str(result['response'])
                    score = {
                        "id": result['id'],
                        "appearance_score": scores[0],
                        "intrinsic_score": scores[1],
                        "relationship_score": scores[2],
                        "overall_score": scores[3]
                    }
                    score_dict[result['id']] = score
        else:
            continue

        if '_stage_2' in basename:
            score_file = os.path.join(result_dir, f"{basename[:basename.find('_stage_2-result.jsonl')]}-result-score.jsonl")
        else:
            score_file = os.path.join(result_dir, f"{basename[:basename.find('-result.jsonl')]}-result-score.jsonl")
        with open(score_file, 'w+', encoding='utf-8') as f:
            for _, value in score_dict.items():
                f.write(json.dumps(value, ensure_ascii=False) + '\n')
    
