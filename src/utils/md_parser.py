import json
import copy
import numpy as np
from tqdm import tqdm
from difflib import SequenceMatcher


structure_template = {
    "Structure Information": {
        "Intrinsic Attributes": None,
        "Relationship Attributes": None,
    },
    "Questions": {
        "Appearance Quality Questions": None,
        "Intrinsic Attribute Consistency Questions": None,
        "Relationship Attribute Consistency Questions": None,
    },
    "Image Caption": None,
    "Answers": {
        "Appearance Quality Questions": None,
        "Intrinsic Attribute Consistency Questions": None,
        "Relationship Attribute Consistency Questions": None,
    },
    "Evaluation": {
        "Appearance Quality Answers": None,
        "Intrinsic Attribute Consistency Answers": None,
        "Relationship Attribute Consistency Answers": None,
        "Overall Evaluation": {
            "Appearance Quality Summary": None,
            "Intrinsic Attribute Consistency Summary": None,
            "Relationship Attribute Consistency Summary": None,
            "Overall Score": None
        },
    },
}


question_attributes = [
    "entities",
    "answer",
    "explanation",
    "score",
    "manual_score"
]


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


def _get_best_match(source, target, path: str = None, mismatch_log: dict = None):
    # calculate sequence similarity matrix
    sim_matrix = np.zeros([len(source), len(target)])
    for i, src_key in enumerate(source):
        for j, tgt_key in enumerate(target):
            sim_matrix[i][j] = SequenceMatcher(
                None, a=src_key.strip().lower(), b=tgt_key.strip().lower()
            ).quick_ratio()

    # assign match result
    src_best_match_indices = np.argmax(sim_matrix, axis=1)
    tgt_best_match_indices = np.argmax(sim_matrix, axis=0)
    src_match = [(i, tgt) for i, tgt in enumerate(src_best_match_indices)]
    tgt_match = [(src, i) for i, src in enumerate(tgt_best_match_indices)]
    match_pairs = list(set(src_match) & set(tgt_match))
    match_pairs = sorted(match_pairs, key=lambda s: s[1])

    # convert result to text format
    match_result = []
    for src_idx, tgt_idx in match_pairs:
        src_key = source[src_idx]
        tgt_key = target[tgt_idx]
        # log imperfect match
        if sim_matrix[src_idx][tgt_idx] < 0.9 and mismatch_log is not None:
            mismatch_log["imperfect_match"].append(
                {
                    "path": path,
                    "src_key": src_key,
                    "tgt_key": tgt_key,
                    "similarity": sim_matrix[src_idx][tgt_idx],
                }
            )
            mismatch_log["error"] = True
        match_result.append([src_key, tgt_key])
    return match_result


def _get_data_from_struct_by_key_chain(struct, key_chain):
    for key in key_chain:
        if key in struct:
            struct = struct[key]
        else:
            return None
    return struct


def _set_data_for_struct_by_key_chain(struct, key_chain, value):
    for key in key_chain[:-1]:
        struct = struct[key]
    struct[key_chain[-1]] = value
    
    
def _parse_question_attributes(attribute_list: list) -> dict:
    attribute_mapper = {item.split(':')[0].strip(): ':'.join(item.split(':')[1:]).strip() for item in attribute_list}
    best_match = _get_best_match(source=list(attribute_mapper.keys()), target=question_attributes)
    cleaned_attributes = {tgt_key: attribute_mapper[src_key] for src_key, tgt_key in best_match}
    if 'score' in cleaned_attributes:
        cleaned_attributes['score'] = extract_score_from_str(cleaned_attributes['score'])
    return cleaned_attributes


def _get_valid_question_map(questions, check_fn = None, clean_key: bool = True):
    if check_fn is None:
        def check_fn(item):
            return isinstance(item, str)
        
    question_map = {}
    for i, item in enumerate(questions):
        if check_fn(item):
            question_map[item.strip()] = None
        else:
            if not isinstance(item, str):
                assert isinstance(item, list)
                if i != 0 and check_fn(questions[i - 1]):
                    question_map[questions[i - 1].strip()] = _parse_question_attributes(item)
    cleaned_question_map = {':'.join(key.split(':')[1:]).strip() if clean_key else key: value for key, value in question_map.items()}
    return cleaned_question_map


def _match_overall_evaluation(raw_evaluation: list):
    def is_str(item):
        return isinstance(item, str)
    
    eval_map = _get_valid_question_map(questions=raw_evaluation, check_fn=is_str, clean_key=False)
    new_eval_map = {}
    for key, value in eval_map.items():
        if key.startswith('Attribute') or key.startswith('attribute'):
            new_eval_map[f"Intrinsic {key}"] = value
        else:
            new_eval_map[key] = value
    return new_eval_map


def _get_path_from_key_chain(key_chain):
    path = "root"
    for key in key_chain:
        path += f" -> {key}"
    return path


def parse_structured_data(
    structured_data: dict,
    target_structure: dict,
    file: str = None,
    verbose: bool = False,
    match_questions: bool = True,
    strict_questions: bool = True,
    force_struct_info: bool = True
):
    target_structure = copy.deepcopy(target_structure)
    mismatch_log = {
        "file": file,
        "missed_keys": {},
        "redundant_keys": [],
        "imperfect_match": [],
        "error": False,
    }
    
    def _clean_keys(struct: dict):
        new_struct = {}
        for key, value in struct.items():
            if ":" not in key and "." not in key:
                new_struct[key] = value
            else:
                if ":" in key:
                    new_key = key.split(":")[-1].strip()
                elif "." in key:
                    new_key = key.split(".")[-1].strip()
                new_struct[new_key] = value
        return new_struct

    def _match_keys(source, target, path: str = "root"):
        try:
            source_keys = list(source.keys())
        except AttributeError:
            raise AttributeError(f"[!] Invalid data: {source} for dict operations")
        target_keys = list(target.keys())

        best_match = _get_best_match(source=source_keys, target=target_keys, path=path, mismatch_log=mismatch_log)

        if path == "root" and force_struct_info:
            assert (
                best_match[0][1] == "Structure Information"
            ), f"invalid data (missed 'Structure Information')"

        # handle matched keys
        for src_key, tgt_key in best_match:
            if isinstance(target[tgt_key], dict) and tgt_key != "Overall Evaluation":
                _match_keys(
                    source=source[src_key],
                    target=target[tgt_key],
                    path=f"{path} -> {src_key}",
                )
            elif isinstance(target[tgt_key], dict) and tgt_key == "Overall Evaluation":
                matched_eval = _match_overall_evaluation(source[src_key])
                _match_keys(
                    source=matched_eval,
                    target=target[tgt_key],
                    path=f"{path} -> {src_key}"
                )
            else:
                target[tgt_key] = source[src_key]

            if path == "root" and tgt_key == "Structure Information":
                assert "Intrinsic Attributes" in target[tgt_key] and isinstance(
                    target[tgt_key]["Intrinsic Attributes"], dict
                ), f"invalid data (missed 'Intrinsic Attributes')"
                target[tgt_key]["Intrinsic Attributes"] = _clean_keys(
                    target[tgt_key]["Intrinsic Attributes"]
                )
                entity_dict = {
                    key: None
                    for key in list(target[tgt_key]["Intrinsic Attributes"].keys())
                }
                if "Questions" in target:
                    target["Questions"]["Appearance Quality Questions"] = copy.deepcopy(
                        entity_dict
                    )
                    target["Questions"]["Intrinsic Attribute Consistency Questions"] = (
                        copy.deepcopy(entity_dict)
                    )
                if "Image Caption" in target:
                    target["Image Caption"] = copy.deepcopy(entity_dict)
                if "Answers" in target:
                    target["Answers"]["Appearance Quality Questions"] = copy.deepcopy(
                        entity_dict
                    )
                    target["Answers"]["Intrinsic Attribute Consistency Questions"] = (
                        copy.deepcopy(entity_dict)
                    )
                if "Evaluation" in target:
                    target["Evaluation"]["Appearance Quality Answers"] = copy.deepcopy(
                        entity_dict
                    )
                    target["Evaluation"]["Intrinsic Attribute Consistency Answers"] = (
                        copy.deepcopy(entity_dict)
                    )

        # handle missed & redundant keys
        missed_keys = list(
            set(target_keys) - set([tgt_key for _, tgt_key in best_match])
        )
        redundant_keys = list(
            set(source_keys) - set([src_key for src_key, _ in best_match])
        )

        if "Image Caption" not in path.title():
            for key in missed_keys:
                value = f"No {key.lower()}"
                target[key] = value
                mismatch_log["missed_keys"][f"{path} -> {key}"] = value
        else:
            for key in missed_keys:
                target.pop(key)

        mismatch_log["redundant_keys"].extend(
            [f"{path} -> {key}" for key in redundant_keys]
        )

        if (len(missed_keys) != 0 and "Image Caption" not in path.title()) or len(
            redundant_keys
        ) != 0:
            mismatch_log["error"] = True

    def _match_questions(structured_data):
        def is_valid_question(string):
            if not isinstance(string, str):
                return False
            string = string.strip()
            if (
                string.startswith("question") or string.startswith("Question")
            ) and "?" in string:
                return True
            else:
                return False

        def _match_questions_single(
            struct, question_key_chain, answer_key_chain, evaluation_key_chain
        ):
            raw_questions = _get_data_from_struct_by_key_chain(
                struct=struct, key_chain=question_key_chain
            )
            raw_answers = _get_data_from_struct_by_key_chain(
                struct=struct, key_chain=answer_key_chain
            )
            raw_evaluations = _get_data_from_struct_by_key_chain(
                struct=struct, key_chain=evaluation_key_chain
            )
            if not (
                isinstance(raw_questions, list)
                and isinstance(raw_answers, list)
                and isinstance(raw_evaluations, list)
            ):
                return [], [], []

            valid_questions = _get_valid_question_map(questions=raw_questions, check_fn=is_valid_question if strict_questions else None)
            valid_answers = _get_valid_question_map(questions=raw_answers, check_fn=is_valid_question if strict_questions else None)
            valid_evaluations = _get_valid_question_map(questions=raw_evaluations, check_fn=is_valid_question if strict_questions else None)
            if valid_questions == {} or valid_answers == {} or valid_evaluations == {}:
                return [], [], []

            answer_match_result = _get_best_match(
                source=list(valid_answers.keys()),
                target=list(valid_questions.keys()),
                path=_get_path_from_key_chain(answer_key_chain),
                mismatch_log=mismatch_log
            )
            evaluation_match_result = _get_best_match(
                source=list(valid_evaluations.keys()),
                target=list(valid_questions.keys()),
                path=_get_path_from_key_chain(evaluation_key_chain),
                mismatch_log=mismatch_log
            )

            # handle mismatch
            if [question for _, question in answer_match_result] != [
                question for _, question in evaluation_match_result
            ]:
                answer_q = [question for _, question in answer_match_result]
                evaluation_q = [question for _, question in evaluation_match_result]
                common_q = list(set(answer_q) & set(evaluation_q))
                new_answer_match_result = []
                for item in answer_match_result:
                    if item[1] in common_q:
                        new_answer_match_result.append(item)
                new_evaluation_match_result = []
                for item in evaluation_match_result:
                    if item[1] in common_q:
                        new_evaluation_match_result.append(item)
                answer_match_result = new_answer_match_result
                evaluation_match_result = new_evaluation_match_result

            matched_questions = []
            for question in [_question for _, _question in answer_match_result]:
                matched_questions.append(
                    {"question": question, "value": valid_questions[question]}
                )
            matched_answers = []
            for question_a, question in answer_match_result:
                matched_answers.append(
                    {"question": question, "value": valid_answers[question_a]}
                )
            matched_evaluations = []
            for question_e, question in evaluation_match_result:
                matched_evaluations.append(
                    {"question": question, "value": valid_evaluations[question_e]}
                )

            return matched_questions, matched_answers, matched_evaluations

        def _extract_questions_single(
            struct, question_key_chain
        ):
            raw_questions = _get_data_from_struct_by_key_chain(
                struct=struct, key_chain=question_key_chain
            )
            if not (
                isinstance(raw_questions, list)
            ):
                return []

            valid_questions = _get_valid_question_map(questions=raw_questions, check_fn=is_valid_question)
            if valid_questions == {}:
                return []

            extracted_questions = []
            for question in valid_questions:
                extracted_questions.append(
                    {"question": question, "value": valid_questions[question]}
                )

            return extracted_questions

        if "Structure Information" in structured_data:
            entities = list(
                structured_data["Structure Information"]["Intrinsic Attributes"].keys()
            )
            for question_type in ["Appearance Quality", "Intrinsic Attribute Consistency"]:
                for entity in entities:
                    question_key_chain = ["Questions", f"{question_type} Questions", entity]
                    answer_key_chain = ["Answers", f"{question_type} Questions", entity]
                    evaluation_key_chain = [
                        "Evaluation",
                        f"{question_type} Answers",
                        entity,
                    ]
                    if "Answers" in structured_data and "Evaluation" in structured_data:
                        questions, answers, evaluations = _match_questions_single(
                            struct=structured_data,
                            question_key_chain=question_key_chain,
                            answer_key_chain=answer_key_chain,
                            evaluation_key_chain=evaluation_key_chain,
                        )
                        _set_data_for_struct_by_key_chain(
                            struct=structured_data,
                            key_chain=question_key_chain,
                            value=questions,
                        )
                        _set_data_for_struct_by_key_chain(
                            struct=structured_data, key_chain=answer_key_chain, value=answers
                        )
                        _set_data_for_struct_by_key_chain(
                            struct=structured_data,
                            key_chain=evaluation_key_chain,
                            value=evaluations,
                        )
                    else:
                        questions = _extract_questions_single(
                            struct=structured_data,
                            question_key_chain=question_key_chain
                        )
                        _set_data_for_struct_by_key_chain(
                            struct=structured_data,
                            key_chain=question_key_chain,
                            value=questions,
                        )
            
            # handle relationship attribute consistency questions
            question_key_chain = [
                "Questions",
                f"Relationship Attribute Consistency Questions",
            ]
            answer_key_chain = ["Answers", f"Relationship Attribute Consistency Questions"]
            evaluation_key_chain = [
                "Evaluation",
                f"Relationship Attribute Consistency Answers",
            ]
            if "Answers" in structured_data and "Evaluation" in structured_data:
                questions, answers, evaluations = _match_questions_single(
                    struct=structured_data,
                    question_key_chain=question_key_chain,
                    answer_key_chain=answer_key_chain,
                    evaluation_key_chain=evaluation_key_chain,
                )
            
                _set_data_for_struct_by_key_chain(
                    struct=structured_data, key_chain=question_key_chain, value=questions
                )
                _set_data_for_struct_by_key_chain(
                    struct=structured_data, key_chain=answer_key_chain, value=answers
                )
                _set_data_for_struct_by_key_chain(
                    struct=structured_data, key_chain=evaluation_key_chain, value=evaluations
                )
            else:
                questions = _extract_questions_single(
                    struct=structured_data,
                    question_key_chain=question_key_chain
                )
                _set_data_for_struct_by_key_chain(
                    struct=structured_data,
                    key_chain=question_key_chain,
                    value=questions,
                )
        else:
            question_key_chain = [list(structured_data.keys())[0]]
            if isinstance(structured_data[question_key_chain[0]], dict):
                question_key_chain.append(list(structured_data[question_key_chain[0]].keys())[0])
            questions = _extract_questions_single(struct=structured_data, question_key_chain=question_key_chain)
            _set_data_for_struct_by_key_chain(
                struct=structured_data, key_chain=question_key_chain, value=questions
            )
            
    _match_keys(source=structured_data, target=target_structure)
    if match_questions:
        _match_questions(structured_data=target_structure)

    if verbose and mismatch_log["error"] is True:
        tqdm.write(s=json.dumps(mismatch_log, ensure_ascii=False, indent=4))

    return target_structure, mismatch_log


def json_to_markdown(
    struct,
    title_level: int = 0,
    list_level: int = 0,
    from_list: bool = False,
    is_overall_eval: bool = False,
    ignore_score: bool = False,
):
    if isinstance(struct, dict) and not is_overall_eval:
        text = ""
        if "question" in struct and "value" in struct:
            if not ignore_score:
                new_struct = (
                    [
                        f"question: {struct['question']}",
                        [f"{key}: {value}" for key, value in struct["value"].items()],
                    ]
                    if struct["value"] is not None
                    else [f"question: {struct['question']}"]
                )
            else:
                new_struct = (
                    [
                        f"question: {struct['question']}",
                        [
                            f"{key}: {value}"
                            for key, value in struct["value"].items()
                            if key != "score" and key != "manual_score"
                        ],
                    ]
                    if struct["value"] is not None
                    else [f"question: {struct['question']}"]
                )
            text += json_to_markdown(
                new_struct,
                title_level=title_level,
                list_level=list_level - 1 if from_list else list_level,
                ignore_score=ignore_score,
            )
        else:
            for key, value in struct.items():
                if value is not None:
                    sub_text = json_to_markdown(
                        value,
                        title_level=title_level + 1,
                        list_level=list_level,
                        ignore_score=ignore_score,
                    )
                    text += f"{'#' * (title_level + 1)} {key}\n"
                    text += sub_text
        return text
    elif isinstance(struct, list):
        text = ""
        for item in struct:
            sub_text = json_to_markdown(
                item,
                title_level=title_level,
                list_level=list_level + 1,
                from_list=True,
                ignore_score=ignore_score,
            )
            if isinstance(item, list) or isinstance(item, dict):
                text += f"{sub_text}"
            else:
                text += f"{'    ' * (list_level)}- {sub_text}\n"
        return text
    elif isinstance(struct, dict) and is_overall_eval:
        new_struct = []
        for key, value in struct.items():
            new_struct.append(key)
            if isinstance(value, dict):
                if not ignore_score:
                    new_struct.append(
                        [f"{_key}: {_value}" for _key, _value in value.items()]
                    )
                else:
                    new_struct.append(
                        [
                            f"{_key}: {_value}"
                            for _key, _value in value.items()
                            if _key != "score" and key != "manual_score"
                        ]
                    )
            else:
                new_struct.append([f"explanation: N/A", f"score: N/A"])
        return json_to_markdown(new_struct, ignore_score=ignore_score)
    else:
        return str(struct)
