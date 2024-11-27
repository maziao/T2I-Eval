import os
import json
import copy
import markdown_to_json
from tqdm import tqdm, trange
from abc import abstractmethod
from typing import List, Dict, Optional


from src.utils.md_parser import parse_structured_data, json_to_markdown
from src.utils.extract_scores import (
    extract_score_from_str,
    extract_score_list_from_str
)
from src.prompt import (
    EXTRACT_TEMPLATE,
    REF_FREE_APPEARANCE_ANSWER_TEMPLATE,
    REF_FREE_APPEARANCE_ANSWER_TEMPLATE_STAGE_1,
    REF_FREE_APPEARANCE_ANSWER_TEMPLATE_STAGE_2,
    REF_BASED_APPEARANCE_ANSWER_TEMPLATE,
    REF_BASED_APPEARANCE_ANSWER_TEMPLATE_STAGE_1,
    REF_BASED_APPEARANCE_ANSWER_TEMPLATE_STAGE_2,
    INTRINSIC_ANSWER_TEMPLATE,
    RELATIONSHIP_ANSWER_TEMPLATE,
    INTRINSIC_EVAL_TEMPLATE,
    INTRINSIC_EVAL_TEMPLATE_STAGE_1,
    INTRINSIC_EVAL_TEMPLATE_STAGE_2,
    RELATIONSHIP_EVAL_TEMPLATE,
    RELATIONSHIP_EVAL_TEMPLATE_STAGE_1,
    RELATIONSHIP_EVAL_TEMPLATE_STAGE_2,
    OVERALL_SUMMARIZE_TEMPLATE,
    OVERALL_SUMMARIZE_TEMPLATE_STAGE_1,
    OVERALL_SUMMARIZE_TEMPLATE_STAGE_2,
    APPEARANCE_SUMMARIZE_TEMPLATE,
    APPEARANCE_SUMMARIZE_TEMPLATE_STAGE_1,
    APPEARANCE_SUMMARIZE_TEMPLATE_STAGE_2,
    INTRINSIC_SUMMARIZE_TEMPLATE,
    INTRINSIC_SUMMARIZE_TEMPLATE_STAGE_1,
    INTRINSIC_SUMMARIZE_TEMPLATE_STAGE_2,
    RELATIONSHIP_SUMMARIZE_TEMPLATE,
    RELATIONSHIP_SUMMARIZE_TEMPLATE_STAGE_1,
    RELATIONSHIP_SUMMARIZE_TEMPLATE_STAGE_2,
    MERGE_SUMMARIZE_TEMPLATE,
    MERGE_SUMMARIZE_TEMPLATE_STAGE_1,
    MERGE_SUMMARIZE_TEMPLATE_STAGE_2,
    REF_BASED_ANSWER_TEMPLATE_ABLATION_1,
    REF_FREE_ANSWER_TEMPLATE_ABLATION_1,
    EVAL_TEMPLATE_ABLATION_1,
    REF_BASED_APPEARANCE_ANSWER_TEMPLATE_ABLATION_2,
    REF_FREE_APPEARANCE_ANSWER_TEMPLATE_ABLATION_2,
    INTRINSIC_ANSWER_TEMPLATE_ABLATION_2,
    INTRINSIC_EVAL_TEMPLATE_ABLATION_2,
    RELATIONSHIP_ANSWER_TEMPLATE_ABLATION_2,
    RELATIONSHIP_EVAL_TEMPLATE_ABLATION_2
)


EXTRACT_STRUCTURE_TEMPLATE = {
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
}

OVERALL_STRUCTURE_TEMPLATE = {
    "Overall Evaluation": {
        "Appearance Quality Summary": None,
        "Intrinsic Attribute Consistency Summary": None,
        "Relationship Attribute Consistency Summary": None,
        "Overall Score": None
    },
}


ANSWER_PROMPT = {
    "appearance": REF_FREE_APPEARANCE_ANSWER_TEMPLATE,
    "appearance - stage_1": REF_FREE_APPEARANCE_ANSWER_TEMPLATE_STAGE_1,
    "appearance - stage_2": REF_FREE_APPEARANCE_ANSWER_TEMPLATE_STAGE_2,
    "appearance + ref": REF_BASED_APPEARANCE_ANSWER_TEMPLATE,
    "appearance + ref - stage_1": REF_BASED_APPEARANCE_ANSWER_TEMPLATE_STAGE_1,
    "appearance + ref - stage_2": REF_BASED_APPEARANCE_ANSWER_TEMPLATE_STAGE_2,
    "intrinsic": INTRINSIC_ANSWER_TEMPLATE,
    "relationship": RELATIONSHIP_ANSWER_TEMPLATE ,
}


EVALUATION_PROMPT = {
    "intrinsic": INTRINSIC_EVAL_TEMPLATE,
    "intrinsic - stage_1": INTRINSIC_EVAL_TEMPLATE_STAGE_1,
    "intrinsic - stage_2": INTRINSIC_EVAL_TEMPLATE_STAGE_2,
    "relationship": RELATIONSHIP_EVAL_TEMPLATE,
    "relationship - stage_1": RELATIONSHIP_EVAL_TEMPLATE_STAGE_1,
    "relationship - stage_2": RELATIONSHIP_EVAL_TEMPLATE_STAGE_2,
}


SUMMARIZE_PROMPT = {
    "appearance": APPEARANCE_SUMMARIZE_TEMPLATE,
    "appearance - stage_1": APPEARANCE_SUMMARIZE_TEMPLATE_STAGE_1,
    "appearance - stage_2": APPEARANCE_SUMMARIZE_TEMPLATE_STAGE_2,
    "intrinsic": INTRINSIC_SUMMARIZE_TEMPLATE,
    "intrinsic - stage_1": INTRINSIC_SUMMARIZE_TEMPLATE_STAGE_1,
    "intrinsic - stage_2": INTRINSIC_SUMMARIZE_TEMPLATE_STAGE_2,
    "relationship": RELATIONSHIP_SUMMARIZE_TEMPLATE,
    "relationship - stage_1": RELATIONSHIP_SUMMARIZE_TEMPLATE_STAGE_1,
    "relationship - stage_2": RELATIONSHIP_SUMMARIZE_TEMPLATE_STAGE_2,
}


ABLATION_2_ANSWER_PROMPT = {
    "appearance": REF_FREE_APPEARANCE_ANSWER_TEMPLATE_ABLATION_2,
    "appearance + ref": REF_BASED_APPEARANCE_ANSWER_TEMPLATE_ABLATION_2,
    "intrinsic": INTRINSIC_ANSWER_TEMPLATE_ABLATION_2,
    "relationship": RELATIONSHIP_ANSWER_TEMPLATE_ABLATION_2,
}


ABLATION_2_EVAL_PROMPT = {
    "intrinsic": INTRINSIC_EVAL_TEMPLATE_ABLATION_2,
    "relationship": RELATIONSHIP_EVAL_TEMPLATE_ABLATION_2,
}


category_long_to_short = {
    "Appearance Quality": "appearance",
    "Intrinsic Attribute Consistency": "intrinsic",
    "Relationship Attribute Consistency": "relationship",
}


def delete_title(text: str):
    """Delete titles and reserve question only for generated text in answer and evaluation stages.

    Args:
        text (str): generated text in answer or evaluation stage

    Returns:
        str: reformatted text
    """
    splits = text.split("\n")
    non_title_splits = []
    for split in splits:
        if (not split.startswith("#")) and split != "":
            non_title_splits.append(split)
    return "\n".join(non_title_splits)


def add_line_sep_before_title(text: str):
    """Add line separator '\\n' before titles in markdown-formatted string to construct legal markdown text.
    Separator will not be added if there is one in the corresponding place.

    Args:
        text (str): generated text in markdown format

    Returns:
        str: reformatted text
    """
    splits = text.split("#")
    for i in range(len(splits)):
        if splits[i] != "" and not splits[i].endswith("\n"):
            splits[i] += "\n"
    return "#".join(splits)


class InferenceEngine:
    def __init__(
        self,
        data_file: str,
        image_root: str,
        output_dir: str,
        max_retry: int = 0,
        model_init_kwargs: dict = {}
    ) -> None:
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        with open(data_file, "r+", encoding="utf-8") as f:
            self.dataset = json.load(f)
        
        self.image_root = image_root
        
        for sample in self.dataset:
            sample['gt_image'] = os.path.join(self.image_root, sample['gt_image'])
            if sample['ref_image'] is not None:
                sample['ref_image'] = os.path.join(self.image_root, sample['ref_image'])
        
        self.categories_answer = [
            "appearance_answer",
            "appearance_answer_stage_1",
            "appearance_answer_stage_2",
            "intrinsic_answer",
            "relationship_answer",
            "all_in_one_answer"
        ]
        self.categories_eval = [
            "intrinsic_eval",
            "intrinsic_eval_stage_1",
            "intrinsic_eval_stage_2",
            "relationship_eval",
            "relationship_eval_stage_1",
            "relationship_eval_stage_2",
            "all_in_one_eval"
        ]
        self.categories_aspect_summary = [
            "appearance_summary",
            "appearance_summary_stage_1",
            "appearance_summary_stage_2",
            "intrinsic_summary",
            "intrinsic_summary_stage_1",
            "intrinsic_summary_stage_2",
            "relationship_summary",
            "relationship_summary_stage_1",
            "relationship_summary_stage_2",
        ]
        self.categories_overall_summary = [
            "summarize",
            "summarize_stage_1",
            "summarize_stage_2"
        ]
        self.stages = ["extract"] + self.categories_answer + self.categories_eval + self.categories_aspect_summary + self.categories_overall_summary
        
        self.output_file_mapper = {
            stage: os.path.join(output_dir, f"{stage}-result.jsonl")
            for stage in self.stages
        }
        
        self.cache_file = {
            'fine_grained': os.path.join(output_dir, f"fine_grained_task_cache.jsonl"),
            'coarse_grained': os.path.join(output_dir, f"coarse_grained_task_cache.jsonl"),
        }
        
        # get completed ids for each stage
        self.progress_map = {}
        for stage in self.stages:
            if os.path.exists(self.output_file_mapper[stage]):
                with open(self.output_file_mapper[stage], "r+", encoding="utf-8") as f:
                    lines = f.readlines()
                    single_stage_results = [json.loads(line) for line in lines]
                    self.progress_map[stage] = {
                        result["id"]: result for result in single_stage_results
                    }
                del lines
            else:
                self.progress_map[stage] = {}
                
        self.output_mapper = {stage: [] for stage in self.stages}
        
        assert max_retry >= 0
        self.max_retry = max_retry
        self.orig_image_placeholder = '<ImagePlaceholder>'
        
        self.init_model(**model_init_kwargs)
        
    @abstractmethod
    def init_model(self, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def replace_image_placeholder(self, text: str) -> str:
        raise NotImplementedError
    
    @abstractmethod
    def chat_single_round(self, prompt: str, gt_image: str = None, ref_image: str = None, history = None, retry: bool = False) -> tuple:
        raise NotImplementedError
    
    def inference(self, granularity: str, multi_stage: bool = True, first_stage_orig: bool = False, fine_grained_do_summarize: bool = False, separate_aspects: bool = True, simple_answer_and_eval: bool = True, coarse_grained_skip_summarize: bool = False, ablation: int = None):
        assert granularity in ['fine', 'coarse']
        
        if multi_stage and first_stage_orig:
            tqdm.write(f"[!] Performing inference with explanation and scoring separated, and use original prompt template for explanation.[!]")
            ANSWER_PROMPT["appearance - stage_1"] = REF_FREE_APPEARANCE_ANSWER_TEMPLATE
            ANSWER_PROMPT["appearance + ref - stage_1"] = REF_BASED_APPEARANCE_ANSWER_TEMPLATE
            EVALUATION_PROMPT["intrinsic - stage_1"] = INTRINSIC_EVAL_TEMPLATE
            EVALUATION_PROMPT["relationship - stage_1"] = RELATIONSHIP_EVAL_TEMPLATE
        elif multi_stage:
            tqdm.write(f"[!] Performing inference with explanation and scoring separated.[!]")
         
        if granularity == 'fine':
            for i in trange(len(self.dataset)):
                self.fine_grained_pipeline(sample_index=i, multi_stage=multi_stage, do_summarize=fine_grained_do_summarize, separate_aspects=separate_aspects, simple_answer_and_eval=simple_answer_and_eval)
                self.dump_cache_to_file()
        else:
            for i in trange(len(self.dataset)):
                self.coarse_grained_pipeline(sample_index=i, multi_stage=multi_stage, separate_aspects=separate_aspects, simple_answer_and_eval=simple_answer_and_eval, skip_summarize=coarse_grained_skip_summarize, ablation=ablation)
                self.dump_cache_to_file()
        
        if multi_stage and first_stage_orig:
            tqdm.write(f"[!] Reset prompt template for explanation.[!]")
            ANSWER_PROMPT["appearance - stage_1"] = REF_FREE_APPEARANCE_ANSWER_TEMPLATE_STAGE_1
            ANSWER_PROMPT["appearance + ref - stage_1"] = REF_BASED_APPEARANCE_ANSWER_TEMPLATE_STAGE_1
            EVALUATION_PROMPT["intrinsic - stage_1"] = INTRINSIC_EVAL_TEMPLATE_STAGE_1
            EVALUATION_PROMPT["relationship - stage_1"] = RELATIONSHIP_EVAL_TEMPLATE_STAGE_1
    
    def dump_cache_to_file(self):
        for key in self.output_mapper:
            if len(self.output_mapper[key]) == 0:
                continue
            assert key in self.output_file_mapper
            with open(self.output_file_mapper[key], "a+", encoding='utf-8') as f:
                for line in self.output_mapper[key]:
                    f.write(line)
            self.output_mapper[key] = []
    
    def fine_grained_pipeline(self, sample_index: int, multi_stage: bool = False, do_summarize: bool = False, separate_aspects: bool = False, simple_answer_and_eval: bool = False):
        sample = self.dataset[sample_index]
        tqdm.write(f"# Performing inference for sample {sample['id']}")
        
        question_map = {
            "appearance": [{"id": None, **single_question} for single_question in sample['appearance_questions']],
            "intrinsic": [{"id": None, **single_question} for single_question in sample['intrinsic_questions']],
            "relationship": [{"id": None, **single_question} for single_question in sample['relationship_questions']]
        }
        for category in question_map:
            for i in range(len(question_map[category])):
                question_map[category][i]['id'] = f"{sample_index}-{i}"
                question_map[category][i]['value'] = {}
        
        extract_response_structured = sample['structured_info_str']
        
        # stage 2 & 3: answer & eval
        evaluation_map = self.answer_and_eval_stage(
            question_map=question_map,
            extract_response_structured=extract_response_structured,
            gt_image=sample['gt_image'],
            ref_image=sample['ref_image'],
            multi_stage=multi_stage,
            simple_answer_and_eval=simple_answer_and_eval
        )
        if do_summarize:
            # stage 4: summarize
            summary = self.summarize_stage(
                gt_image=sample['gt_image'],
                structure_info=extract_response_structured,
                evaluation_map=evaluation_map,
                sample_index=sample_index,
                multi_stage=multi_stage,
                separate_aspects=separate_aspects
            )
            return summary is not None
        return evaluation_map is not None
    
    def coarse_grained_pipeline(self, sample_index: int, multi_stage: bool = False, separate_aspects: bool = False, simple_answer_and_eval: bool = False, skip_summarize: bool = False, ablation: int = None):
        sample = self.dataset[sample_index]
        tqdm.write(f"# Performing inference for sample {sample['id']}")
        
        # stage 1: extract -> `matched_structured_data`, `qmap`
        question_map, extract_response_structured = self.extract_stage(
            image_caption=sample['image_caption'],
            gt_image=sample['gt_image'],
            sample_index=sample_index
        )
        
        # stage 2 & 3: answer & eval
        evaluation_map = self.answer_and_eval_stage(
            question_map=question_map,
            extract_response_structured=extract_response_structured,
            gt_image=sample['gt_image'],
            ref_image=sample['ref_image'],
            multi_stage=multi_stage,
            simple_answer_and_eval=simple_answer_and_eval,
            ablation=ablation,
            sample_index=sample_index
        )
        
        if not skip_summarize:
            # stage 4: summarize
            summary = self.summarize_stage(
                gt_image=sample['gt_image'],
                structure_info=extract_response_structured,
                evaluation_map=evaluation_map,
                sample_index=sample_index,
                multi_stage=multi_stage,
                separate_aspects=separate_aspects,
                ablation=ablation
            )
            return summary is not None
        else:
            return evaluation_map is not None
    
    def extract_stage(self, image_caption: str, gt_image: str, sample_index: int) -> tuple:
        if sample_index in self.progress_map['extract'] and self.progress_map['extract'][sample_index]['questions'] is not None: # `self.progress_map['extract']['questions'] is not None` may be redundant
            extract_output = self.progress_map["extract"][sample_index]
            tqdm.write(f"  stage 1 (extract): using cached result")
        else:
            # 1.1 construct sample, generate response and parse
            extract_output = self._extract_core(image_caption=image_caption, gt_image=gt_image)
            
            # 1.2 save result whether or not error occured
            extract_output['id'] = sample_index
            if not extract_output['error']:
                # construct question index
                for key in extract_output['questions']:
                    for i in range(len(extract_output['questions'][key])):
                        extract_output['questions'][key][i]['id'] = f"{sample_index}-{i}"
                
                self.output_mapper["extract"].append(json.dumps(obj=extract_output, ensure_ascii=False) + "\n")
                tqdm.write(f"  stage 1 (extract): generating and parsing completed")
            else:
                if "extract-error" not in self.output_file_mapper:
                    error_file = f"{self.output_file_mapper['extract'][:self.output_file_mapper['extract'].find('-result.jsonl')]}-error-result.jsonl"
                    self.output_file_mapper["extract-error"] = error_file
                    self.output_mapper["extract-error"] = []
                self.output_mapper["extract-error"].append(json.dumps(obj=extract_output, ensure_ascii=False) + "\n")
                
                tqdm.write(f"  stage 1 (extract): parsing error confronted (sample {sample_index}, skip.\nRaw generation:\n{extract_output['response']}\n[!]")
                
        return extract_output['questions'], extract_output['structured_response']
        
    def _extract_core(self, image_caption: str, gt_image: str):
        # fill in extract prompt and replace <ImagePlaceholder>
        _extract_prompt = self.replace_image_placeholder(
            text=EXTRACT_TEMPLATE.format(text_prompt=image_caption)
        )
        
        # generation & retry loop
        success = False
        retry = 0
        while not success and retry <= self.max_retry:
            extract_response, _ = self.chat_single_round(
                prompt=_extract_prompt,
                gt_image=gt_image,
                ref_image=None,
                history=None,
                retry=retry != 0
            )
            extract_response = add_line_sep_before_title(extract_response)
            extract_response_structured = markdown_to_json.dictify(extract_response)

            # handle illegal output format
            try:
                extract_response_structured, _ = parse_structured_data(
                    structured_data=extract_response_structured, target_structure=EXTRACT_STRUCTURE_TEMPLATE, strict_questions=False
                )
                success = True
            except Exception:
                retry += 1
        
        if success:
            return {
                "id": None,
                "gt_image": gt_image,
                "ref_image": None,
                "query": _extract_prompt,
                "response": extract_response,
                "history": [],
                "structured_response": extract_response_structured,
                "questions": {
                    "appearance": [{"id": None, **single_question, "entity": entity} for entity, q_list in extract_response_structured["Questions"]["Appearance Quality Questions"].items() for single_question in q_list],
                    "intrinsic": [{"id": None, **single_question, "entity": entity} for entity, q_list in extract_response_structured["Questions"]["Intrinsic Attribute Consistency Questions"].items() for single_question in q_list],
                    "relationship": [{"id": None, **single_question} for single_question in extract_response_structured["Questions"]["Relationship Attribute Consistency Questions"]]
                },
                "error": False
            }
        else:
            return {
                "id": None,
                "gt_image": gt_image,
                "ref_image": None,
                "query": _extract_prompt,
                "response": extract_response,
                "history": [],
                "structured_response": None,
                "questions": None,
                "error": True
            }
    
    def answer_and_eval_stage(self, question_map: dict, extract_response_structured: dict, gt_image: str, ref_image: str = None, multi_stage: bool = False, simple_answer_and_eval: bool = False, ablation: int = None, sample_index: int = None):
        tqdm.write(f"  stage 2 & 3 (answer & eval):")
        tqdm.write(f"    Question statistics:")
        for key, value in question_map.items():
            tqdm.write(f"      {key} questions: {len(value)}")
            
        evaluation_map = {"appearance": [], "intrinsic": [], "relationship": []}
        
        if ablation == 1:
            if sample_index in self.progress_map[f"all_in_one_eval"].keys():
                eval_output = self.progress_map[f"all_in_one_eval"][sample_index]
                tqdm.write(f"    {sample_index} (all-in-one answer & eval): using cached result")
            else:
                answer_output, answer_history = self._answer_core_ablation_1(
                    question_map=question_map,
                    gt_image=gt_image,
                    ref_image=ref_image,
                    sample_index=sample_index
                )
                eval_output = self._eval_core_ablation_1(
                    answer_output=answer_output,
                    structure_info=extract_response_structured,
                    gt_image=gt_image,
                    history=answer_history,
                    sample_index=sample_index
                )
                self.output_mapper["all_in_one_eval"].append(json.dumps(obj=eval_output, ensure_ascii=False) + "\n")
                self.output_mapper["all_in_one_answer"].append(json.dumps(obj=answer_output, ensure_ascii=False) + "\n")
                tqdm.write(f"    {sample_index} (all-in-one answer & eval): generating completed")
                
            evaluation_map = {
                "overall": eval_output
            }
            return evaluation_map
        elif ablation == 2:
            for category, question_list in question_map.items():
                if sample_index in self.progress_map[f"{category}_answer"].keys():
                    if category == 'appearance':
                        eval_output = self.progress_map[f"{category}_answer"][sample_index]
                    else:
                        eval_output = self.progress_map[f"{category}_eval"][sample_index]
                    tqdm.write(f"    {sample_index} ({category}): using cached result")
                else:
                    answer_output, answer_history = self._answer_core_ablation_2(
                        question_list=question_list,
                        category=category,
                        gt_image=gt_image,
                        ref_image=ref_image,
                        sample_index=sample_index
                    )
                    if category != 'appearance':
                        eval_output = self._eval_core_ablation_2(
                            answer_output=answer_output,
                            category=category,
                            structure_info=extract_response_structured,
                            gt_image=gt_image,
                            history=answer_history,
                            sample_index=sample_index
                        )
                        self.output_mapper[f"{category}_eval"].append(json.dumps(obj=eval_output, ensure_ascii=False) + "\n")
                    else:
                        eval_output = answer_output
                    
                    self.output_mapper[f"{category}_answer"].append(json.dumps(obj=answer_output, ensure_ascii=False) + "\n")
                    tqdm.write(f"    {sample_index} ({category}): generating completed")
                
                evaluation_map[category] = eval_output
            return evaluation_map
        
        for category, question_list in question_map.items():
            for question in question_list:
                q_id = question['id']
                
                if q_id in self.progress_map[f"{category}_answer"].keys():
                    if category == 'appearance':
                        eval_output = self.progress_map[f"{category}_answer"][q_id]
                    else:
                        eval_output = self.progress_map[f"{category}_eval"][q_id]
                    tqdm.write(f"    {q_id} ({category}): using cached result")
                else:
                    answer_output, answer_output_stage_1, answer_output_stage_2, answer_history = self._answer_core(
                        question=question,
                        category=category,
                        gt_image=gt_image,
                        ref_image=ref_image,
                        multi_stage=multi_stage,
                        simple_format=simple_answer_and_eval
                    )
                    if category != 'appearance':
                        eval_output, eval_output_stage_1, eval_output_stage_2 = self._eval_core(
                            answer_output=answer_output,
                            category=category,
                            structure_info=extract_response_structured,
                            gt_image=gt_image,
                            history=answer_history,
                            multi_stage=multi_stage,
                            simple_format=simple_answer_and_eval
                        )
                        self.output_mapper[f"{category}_eval"].append(json.dumps(obj=eval_output, ensure_ascii=False) + "\n")
                        if multi_stage and eval_output_stage_1 is not None and eval_output_stage_2 is not None:
                            self.output_mapper[f"{category}_eval_stage_1"].append(json.dumps(obj=eval_output_stage_1, ensure_ascii=False) + "\n")
                            self.output_mapper[f"{category}_eval_stage_2"].append(json.dumps(obj=eval_output_stage_2, ensure_ascii=False) + "\n")
                    else:
                        eval_output = answer_output
                    
                    self.output_mapper[f"{category}_answer"].append(json.dumps(obj=answer_output, ensure_ascii=False) + "\n")
                    if multi_stage and answer_output_stage_1 is not None and answer_output_stage_2 is not None:
                        self.output_mapper[f"{category}_answer_stage_1"].append(json.dumps(obj=answer_output_stage_1, ensure_ascii=False) + "\n")
                        self.output_mapper[f"{category}_answer_stage_2"].append(json.dumps(obj=answer_output_stage_2, ensure_ascii=False) + "\n")
                    tqdm.write(f"    {q_id} ({category}): generating completed")

                # append evaluation result to evaluation map
                evaluation_map[category].append(eval_output)
                
        return evaluation_map
        
    def _answer_core(self, question: dict, category: str, gt_image: str, ref_image: str = None, multi_stage: bool = False, simple_format: bool = False):
        answer_prompt_category_1 = f"{category}"
        answer_prompt_category_2 = f"{category}"
        if category == 'appearance':
            if ref_image is not None:
                answer_prompt_category_1 += ' + ref'
                answer_prompt_category_2 += ' + ref'
            # only Appearance Quality Questions may have stage 2 in answering
            if multi_stage:
                answer_prompt_category_1 += ' - stage_1'
                answer_prompt_category_2 += ' - stage_2'
        
        if not simple_format:
            answer_prompt = self.replace_image_placeholder(
                ANSWER_PROMPT[answer_prompt_category_1].format(
                    question=json_to_markdown(struct=question, ignore_score=True)
            ))
            answer_response, history = self.chat_single_round(
                prompt=answer_prompt,
                gt_image=gt_image,
                ref_image=ref_image,
                history=None
            )
            answer_response = add_line_sep_before_title(answer_response)
        else:
            answer_prompt = question['question']
            answer_response, history = self.chat_single_round(
                prompt=answer_prompt,
                gt_image=gt_image,
                history=None
            )
        
        # stage 2 for Appearance Quality Questions
        if multi_stage and category == 'appearance':
            if not simple_format:
                try:
                    answer_response_structured, _ = parse_structured_data(
                        structured_data=json.loads(markdown_to_json.jsonify(answer_response)),
                        target_structure={"Answer": {question['entity']: None}},
                        force_struct_info=False
                    )
                except Exception:
                    answer_response_structured = {'Answer': {question['entity']: [question]}}
            else:
                answer_response_structured = {'Answer': {question['entity']: [{'question': question['question'], 'value': {'explanation': answer_response}}]}}
            answer_score_prompt = self.replace_image_placeholder(
                ANSWER_PROMPT[answer_prompt_category_2].format(
                    question_and_exp=json_to_markdown(struct=answer_response_structured, ignore_score=True)
                )
            )
            answer_score_response, _ = self.chat_single_round(
                prompt=answer_score_prompt,
                gt_image=gt_image,
                ref_image=ref_image,
                history=None
            )
            answer_response_structured['Answer'][question['entity']][0]['value']['score'] = extract_score_from_str(answer_score_response)
        else:
            answer_response_structured = None
            
        kwargs = {
            "id": question['id'],
            "gt_image": gt_image,
            "ref_image": ref_image if category == 'appearance' else None,
            "query": None,
            "response": None,
            "history": []
        }
        if 'entity' in question:
            kwargs['entity'] = question['entity']
        
        # construct output
        answer_output = copy.deepcopy(kwargs)
        answer_output.update({
            "query": answer_prompt,
            "response": json_to_markdown(struct=answer_response_structured) if answer_response_structured is not None and not simple_format else answer_response,
        })
        if simple_format:
            answer_output['question'] = question
        if answer_response_structured is not None:
            answer_output_stage_1 = copy.deepcopy(kwargs)
            answer_output_stage_1.update({
                "query": answer_prompt,
                "response": answer_response,
            })
            answer_output_stage_2 = copy.deepcopy(kwargs)
            answer_output_stage_2.update({
                "query": answer_score_prompt,
                "response": answer_score_response,
                "score": extract_score_from_str(answer_score_response),
            })
        else:
            answer_output_stage_1 = None
            answer_output_stage_2 = None
        return answer_output, answer_output_stage_1, answer_output_stage_2, history
    
    def _answer_core_ablation_1(self, question_map: Dict, gt_image: str, ref_image: Optional[str] = None, sample_index: int = None):
        question_struct = {
            "Appearance Quality Questions": {},
            "Intrinsic Attribute Consistency Questions": {},
            "Relationship Attribute Consistency Questions": question_map['relationship'],
        }
        for question in question_map['appearance']:
            if question['entity'] not in question_struct["Appearance Quality Questions"]:
                question_struct["Appearance Quality Questions"][question['entity']] = [question]
            else:
                question_struct["Appearance Quality Questions"][question['entity']] += [question]
                
        for question in question_map['intrinsic']:
            if question['entity'] not in question_struct["Intrinsic Attribute Consistency Questions"]:
                question_struct["Intrinsic Attribute Consistency Questions"][question['entity']] = [question]
            else:
                question_struct["Intrinsic Attribute Consistency Questions"][question['entity']] += [question]
        
        answer_prompt = self.replace_image_placeholder(
            (
                REF_BASED_ANSWER_TEMPLATE_ABLATION_1
                if ref_image is not None
                else REF_FREE_ANSWER_TEMPLATE_ABLATION_1
            ).format(
                questions=json_to_markdown(
                    struct=question_struct,
                    title_level=1
                )
            )
        )
        answer_response, history = self.chat_single_round(
            prompt=answer_prompt,
            gt_image=gt_image,
            ref_image=ref_image,
            history=None
        )
        answer_response = add_line_sep_before_title(answer_response)
        
        answer_output = {
            "id": sample_index,
            "gt_image": gt_image,
            "ref_image": ref_image,
            "query": answer_prompt,
            "response": answer_response,
            "history": []
        }
        
        return answer_output, history
        
    def _answer_core_ablation_2(self, question_list: List[dict], category: str, gt_image: str, ref_image: str = None, sample_index: int = None):
        answer_prompt_category = f"{category}"
        if category == 'appearance':
            if ref_image is not None:
                answer_prompt_category += ' + ref'

        answer_prompt = self.replace_image_placeholder(
            ABLATION_2_ANSWER_PROMPT[answer_prompt_category].format(questions=json_to_markdown(struct=question_list, ignore_score=True)
        ))
        answer_response, history = self.chat_single_round(
            prompt=answer_prompt,
            gt_image=gt_image,
            ref_image=ref_image,
            history=None
        )
        answer_response = add_line_sep_before_title(answer_response)
        
        answer_output = {
            "id": sample_index,
            "gt_image": gt_image,
            "ref_image": ref_image if category == 'appearance' else None,
            "query": answer_prompt,
            "response": answer_response,
            "history": []
        }

        return answer_output, history
    
    def _eval_core(self, answer_output: dict, category: str, structure_info: dict, gt_image: str, history = None, multi_stage: bool = False, simple_format: bool = False):
        entity = answer_output['entity'] if 'entity' in answer_output else None
        
        eval_prompt_category_1 = f"{category}"
        eval_prompt_category_2 = f"{category} - stage_2"
        if multi_stage:
            eval_prompt_category_1 += ' - stage_1'
        
        if not simple_format:
            eval_prompt = EVALUATION_PROMPT[eval_prompt_category_1].format(
                answer=answer_output['response'],
                structure_info=json_to_markdown(
                    struct=structure_info
                )
            )
        else:
            eval_prompt = f"Give an explanation for the answer according to the image.\nAnswer: {answer_output['response']}"
            
        eval_response, _ = self.chat_single_round(
            prompt=eval_prompt,
            gt_image=gt_image,
            ref_image=None,
            history=history
        )
        eval_response = add_line_sep_before_title(eval_response)
        
        if multi_stage:
            if not simple_format:
                try:
                    eval_response_structured, _ = parse_structured_data(
                        structured_data=json.loads(markdown_to_json.jsonify(eval_response)),
                        target_structure={"Evaluation": {entity: None} if entity is not None else None},
                        force_struct_info=False
                    )
                except Exception:
                    eval_response_structured = {'Evaluation': {entity: [{'question': None, 'value': {}}]}} if entity is not None else {'Evaluation': [{'question': None, 'value': {}}]}
            else:
                question = answer_output['question']
                if not isinstance(question['value'], dict):
                    question['value'] = {}
                question['value']['answer'] = answer_output['response']
                question['value']['explanation'] = eval_response
                eval_response_structured = {'Evaluation': {entity: [question]}} if entity is not None else {'Evaluation': [question]}

            eval_score_prompt = self.replace_image_placeholder(
                EVALUATION_PROMPT[eval_prompt_category_2].format(
                    answer_and_exp=json_to_markdown(
                        struct=eval_response_structured,
                        ignore_score=True
                    ),
                    structure_info=json_to_markdown(
                        struct=structure_info
                    )
                )
            )
            
            eval_score_response, _ = self.chat_single_round(
                prompt=eval_score_prompt,
                gt_image=gt_image,
                ref_image=None,
                history=None
            )

            if entity is not None:
                eval_response_structured['Evaluation'][entity][0]['value']['score'] = extract_score_from_str(eval_score_response)
            else:
                eval_response_structured['Evaluation'][0]['value']['score'] = extract_score_from_str(eval_score_response)
        else:
            eval_response_structured = None
            
        kwargs = {
            "id": answer_output['id'],
            "gt_image": gt_image,
            "ref_image": None,
            "query": None,
            "response": None,
            "history": []
        }
        if 'entity' in answer_output:
            kwargs['entity'] = answer_output['entity']
            
        eval_output = copy.deepcopy(kwargs)
        eval_output.update({
            "query": eval_prompt,
            "response": json_to_markdown(struct=eval_response_structured) if multi_stage else eval_response,
            "history": [
                [
                    answer_output['query'],
                    answer_output['response']
                ]
            ]
        })
        if multi_stage:
            eval_output_stage_1 = copy.deepcopy(kwargs)
            eval_output_stage_1.update({
                "query": eval_prompt,
                "response": eval_response,
                "history": [
                    [
                        answer_output['query'],
                        answer_output['response']
                    ]
                ]
            })
            eval_output_stage_2 = copy.deepcopy(kwargs)
            eval_output_stage_2.update({
                "query": eval_score_prompt,
                "response": eval_score_response,
            })
        else:
            eval_output_stage_1 = None
            eval_output_stage_2 = None
        
        return eval_output, eval_output_stage_1, eval_output_stage_2
    
    def _eval_core_ablation_1(self, answer_output: dict, structure_info: dict, gt_image: str, history = None, sample_index: int = None):
        eval_prompt = EVAL_TEMPLATE_ABLATION_1.format(
            answers=answer_output['response'],
            structure_info=json_to_markdown(
                struct=structure_info
            )
        )
        
        eval_response, _ = self.chat_single_round(
            prompt=eval_prompt,
            gt_image=None,
            ref_image=None
        )
        eval_response = add_line_sep_before_title(eval_response)

        eval_output = {
            "id": sample_index,
            "gt_image": None,
            "ref_image": None,
            "query": eval_prompt,
            "response": eval_response,
            "history": []
        }
        
        return eval_output
    
    def _eval_core_ablation_2(self, answer_output: str, category: str, structure_info: dict, gt_image: str, history = None, sample_index: int = None):
        eval_prompt = ABLATION_2_EVAL_PROMPT[category].format(
            answers=answer_output['response'],
            structure_info=json_to_markdown(
                struct=structure_info
            )
        )
        
        eval_response, _ = self.chat_single_round(
            prompt=eval_prompt,
            gt_image=gt_image,
            ref_image=None
        )
        eval_response = add_line_sep_before_title(eval_response)

        eval_output = {
            "id": sample_index,
            "gt_image": gt_image,
            "ref_image": None,
            "query": eval_prompt,
            "response": eval_response,
            "history": []
        }
        
        return eval_output
    
    def _prepare_evaluations_for_summarize_stage(self, structure_info: dict, evaluation_map: dict, ablation: int = None):
        def strip_title(text: str) -> str:
            splits = text.strip().split('\n')
            if 'Answer' in splits[0] or 'Evaluation' in splits[0]:
                new_text = '\n'.join(splits[1:])
                if text.endswith('\n'):
                    return new_text + '\n'
            else:
                return text
            
        # ablation 1
        if ablation == 1:
            return evaluation_map['overall']['response']
        # ablation 2
        if ablation == 2:
            return {
                "Appearance Quality Answers": strip_title(evaluation_map['appearance']['response']),
                "Intrinsic Attribute Consistency Answers": strip_title(evaluation_map['intrinsic']['response']),
                "Relationship Attribute Consistency Answers": strip_title(evaluation_map['relationship']['response']),
            }
         
        if isinstance(structure_info, dict):
            entity_dict = {
                entity: ""
                for entity in structure_info["Structure Information"][
                    "Intrinsic Attributes"
                ].keys()
            }
        else:
            entity_dict = {
                entity: ""
                for entity in list(set(
                    [evaluation['entity'] for evaluation in evaluation_map["appearance"]]
                    + [evaluation['entity'] for evaluation in evaluation_map["intrinsic"]]
                ))
            }
        evaluations = {
            "Appearance Quality Answers": copy.deepcopy(entity_dict),
            "Intrinsic Attribute Consistency Answers": copy.deepcopy(entity_dict),
            "Relationship Attribute Consistency Answers": "",
        }
        
        for e in evaluation_map["appearance"]:
            eval_text = e["response"]
            eval_text = delete_title(eval_text) + "\n"
            evaluations["Appearance Quality Answers"][e["entity"]] += eval_text

        for e in evaluation_map["intrinsic"]:
            eval_text = e["response"]
            eval_text = delete_title(eval_text) + "\n"
            evaluations["Intrinsic Attribute Consistency Answers"][
                e["entity"]
            ] += eval_text

        for e in evaluation_map["relationship"]:
            eval_text = e["response"]
            eval_text = delete_title(eval_text) + "\n"
            evaluations["Relationship Attribute Consistency Answers"] += eval_text
        return evaluations
    
    def summarize_stage(self, gt_image: str, structure_info: dict, evaluation_map: dict, sample_index: int, multi_stage: bool = False, separate_aspects: bool = False, ablation: int = None):
        if sample_index in self.progress_map["summarize"] and not multi_stage:
            if not separate_aspects:
                output_samples = {category: self.progress_map[category][sample_index] for category in self.categories_overall_summary if "stage" not in category}
            else:
                output_samples = {category: self.progress_map[category][sample_index] for category in self.categories_aspect_summary + self.categories_overall_summary if "stage" not in category}
            tqdm.write(f"  stage 4 (summarize): using cached result")
        elif sample_index in self.progress_map["summarize_stage_2"] and multi_stage:
            if not separate_aspects:
                output_samples = {category: self.progress_map[category][sample_index] for category in self.categories_overall_summary if "stage" in category}
            else:
                output_samples = {category: self.progress_map[category][sample_index] for category in self.categories_aspect_summary + self.categories_overall_summary  if "stage" in category}
            tqdm.write(f"  stage 4 (summarize): using cached result")
        else:
            reformatted_evaluations = self._prepare_evaluations_for_summarize_stage(
                structure_info=structure_info,
                evaluation_map=evaluation_map,
                ablation=ablation
            )
            
            if not separate_aspects:
                output_samples = self._summarize_core(
                    gt_image=gt_image,
                    structure_info=structure_info,
                    evaluations=reformatted_evaluations,
                    multi_stage=multi_stage
                )
            else:
                output_samples = self._summarize_core_separate_aspects(
                    gt_image=gt_image,
                    structure_info=structure_info,
                    evaluations=reformatted_evaluations,
                    multi_stage=multi_stage
                )
            
            for sample_category, sample in output_samples.items():
                sample['id'] = sample_index
                
                self.output_mapper[sample_category].append(json.dumps(obj=sample, ensure_ascii=False) + "\n")
                # with open(self.output_file_mapper[sample_category], "a+", encoding="utf-8") as f:
                #     f.write(json.dumps(obj=sample, ensure_ascii=False) + "\n")
            tqdm.write(f"  stage 4 (summarize): generating completed")
            
        return output_samples
    
    def _summarize_core(self, gt_image: str, structure_info: dict, evaluations: dict, multi_stage: bool = False):
        output_samples = {}
        summarize_prompt = self.replace_image_placeholder(
            text=(OVERALL_SUMMARIZE_TEMPLATE if not multi_stage else OVERALL_SUMMARIZE_TEMPLATE_STAGE_1).format(
                eval_result=json_to_markdown(evaluations),
                structure_info=json_to_markdown(struct=structure_info),
            )
        )
        summarize_response, _ = self.chat_single_round(
            prompt=summarize_prompt,
            gt_image=gt_image,
            ref_image=None,
            history=None
        )
        summarize_response = add_line_sep_before_title(summarize_response)
        summarize_response_structured, _ = parse_structured_data(
            structured_data=json.loads(markdown_to_json.jsonify('## Overall Evaluation\n' + summarize_response)),
            target_structure=OVERALL_STRUCTURE_TEMPLATE,
            match_questions=False,
            force_struct_info=False
        )
        
        sample_category = "summarize" if not multi_stage else "summarize_stage_1"
        output_samples[sample_category] = {
            "id": None,
            "gt_image": gt_image,
            "ref_image": None,
            "query": summarize_prompt,
            "response": summarize_response,
            "history": [],
        }
        if not multi_stage:
            output_samples[sample_category]["scores"] = [value["score"] if "score" in value else None for _, value in summarize_response_structured["Overall Evaluation"].items()]
        
        if multi_stage:
            summarize_score_prompt = self.replace_image_placeholder(
                text=OVERALL_SUMMARIZE_TEMPLATE_STAGE_2.format(
                    eval_result_and_exp=json_to_markdown(evaluations)
                    + "\n# Overall Evaluation\n"
                    + json_to_markdown(
                        summarize_response_structured['Overall Evaluation'],
                        is_overall_eval=True,
                        ignore_score=True
                    ),
                    structure_info=json_to_markdown(struct=structure_info),
                )
            )
            summarize_score_response, _ = self.chat_single_round(
                prompt=summarize_score_prompt,
                gt_image=gt_image,
                ref_image=None,
                history=None
            )
            scores = extract_score_list_from_str(summarize_score_response)
            output_samples["summarize_stage_2"] = {
                "id": None,
                "gt_image": gt_image,
                "ref_image": None,
                "query": summarize_score_prompt,
                "response": summarize_score_response,
                "scores": scores,
                "history": [],
            }
        return output_samples
    
    def _summarize_core_separate_aspects(self, gt_image: str, structure_info: dict, evaluations: dict, multi_stage: bool = False):
        output_samples = {}
        result_dict = {}
        for category in list(category_long_to_short.keys()):
            prompt_category = category_long_to_short[category] if not multi_stage else category_long_to_short[category] + ' - stage_1'
            sample_category = category_long_to_short[category] + '_summary' if not multi_stage else category_long_to_short[category] + '_summary_stage_1'
            category_summarize_prompt = self.replace_image_placeholder(
                text=SUMMARIZE_PROMPT[prompt_category].format(
                    eval_result=json_to_markdown({f"{category} Answers": evaluations[f"{category} Answers"]}),
                    structure_info=json_to_markdown(struct=structure_info),
                )
            )
            category_summarize_response, _ = self.chat_single_round(
                prompt=category_summarize_prompt,
                gt_image=gt_image,
                ref_image=None,
                history=None
            )
            category_summarize_response = add_line_sep_before_title(category_summarize_response)
            category_summarize_response_structured, _ = parse_structured_data(
                structured_data=json.loads(markdown_to_json.jsonify('## Overall Evaluation\n' + category_summarize_response)),
                target_structure={"Overall Evaluation": {f"{category} Summary": None}},
                match_questions=False,
                force_struct_info=False
            )
            
            output_samples[sample_category] = {
                "id": None,
                "gt_image": gt_image,
                "ref_image": None,
                "query": category_summarize_prompt,
                "response": category_summarize_response,
                "history": [],
            }
            if not multi_stage:
                output_samples[sample_category]["score"] = category_summarize_response_structured["Overall Evaluation"][f"{category} Summary"]["score"] if "score" in category_summarize_response_structured["Overall Evaluation"][f"{category} Summary"] else None
            
            if multi_stage:
                prompt_category = category_long_to_short[category] + ' - stage_2'
                sample_category = category_long_to_short[category] + '_summary_stage_2'
                category_score_prompt = self.replace_image_placeholder(
                    text=SUMMARIZE_PROMPT[prompt_category].format(
                        eval_result_and_exp=json_to_markdown({f"{category} Answers": evaluations[f"{category} Answers"]})
                        + "\n# Overall Evaluation\n"
                        + json_to_markdown(
                            category_summarize_response_structured["Overall Evaluation"],
                            is_overall_eval=True,
                            ignore_score=True,
                        ),
                        structure_info=json_to_markdown(struct=structure_info),
                    )
                )
                category_score_response, _ = self.chat_single_round(
                    prompt=category_score_prompt,
                    gt_image=gt_image,
                    ref_image=None,
                    history=None
                )
                score = extract_score_from_str(category_score_response)
                output_samples[sample_category] = {
                    "id": None,
                    "gt_image": gt_image,
                    "ref_image": None,
                    "query": category_score_prompt,
                    "response": category_score_response,
                    "score": score,
                    "history": [],
                }
                category_summarize_response_structured["Overall Evaluation"][f"{category} Summary"]["score"] = score
            
            result_dict[f"{category} Summary"] = category_summarize_response_structured["Overall Evaluation"][f"{category} Summary"]
        
        # merge all aspects
        summarize_prompt = self.replace_image_placeholder(
            text=(MERGE_SUMMARIZE_TEMPLATE if not multi_stage else MERGE_SUMMARIZE_TEMPLATE_STAGE_1).format(
                eval_result=json_to_markdown(evaluations)
                + "\n# Overall Evaluation\n"
                + json_to_markdown(result_dict, is_overall_eval=True),
                structure_info=json_to_markdown(struct=structure_info),
            )
        )
        summarize_response, _ = self.chat_single_round(
            prompt=summarize_prompt,
            gt_image=gt_image,
            ref_image=None,
            history=None
        )
        summarize_response_structured, _ = parse_structured_data(
            structured_data=json.loads(markdown_to_json.jsonify('## Overall Evaluation\n' + summarize_response)),
            target_structure={"Overall Evaluation": {"Overall Score": None}},
            match_questions=False,
            force_struct_info=False
        )
        category = "summarize" if not multi_stage else "summarize_stage_1"
        output_samples[category] = {
            "id": None,
            "gt_image": gt_image,
            "ref_image": None,
            "query": summarize_prompt,
            "response": summarize_response,
            "history": [],
        }
        if not multi_stage:
            output_samples[category]["score"] = summarize_response_structured["Overall Evaluation"]["Overall Score"]["score"] if "score" in summarize_response_structured["Overall Evaluation"]["Overall Score"] else None
        
        if multi_stage:
            summarize_score_prompt = self.replace_image_placeholder(
                text=MERGE_SUMMARIZE_TEMPLATE_STAGE_2.format(
                    eval_result_and_exp=json_to_markdown(evaluations)
                    + "\n# Overall Evaluation\n"
                    + json_to_markdown(
                        result_dict,
                        is_overall_eval=True,
                        ignore_score=False
                    )
                    + json_to_markdown(
                        summarize_response_structured["Overall Evaluation"],
                        is_overall_eval=True,
                        ignore_score=True
                    ),
                    structure_info=json_to_markdown(struct=structure_info),
                )
            )
            summarize_score_response, _ = self.chat_single_round(
                prompt=summarize_score_prompt,
                gt_image=gt_image,
                ref_image=None,
                history=None
            )
            score = extract_score_from_str(summarize_score_response)
            output_samples["summarize_stage_2"] = {
                "id": None,
                "gt_image": gt_image,
                "ref_image": None,
                "query": summarize_score_prompt,
                "response": summarize_score_response,
                "score": score,
                "history": [],
            }
    
        return output_samples
