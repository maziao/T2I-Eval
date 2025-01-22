import os
import json
from tqdm import tqdm
from abc import abstractmethod

from src.utils.md_parser import json_to_markdown
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


category_to_answer_prompt = {
    "Appearance Quality": REF_FREE_APPEARANCE_ANSWER_TEMPLATE,
    "Appearance Quality + Ref": REF_BASED_APPEARANCE_ANSWER_TEMPLATE,
    "Intrinsic Attribute Consistency": INTRINSIC_ANSWER_TEMPLATE,
    "Relationship Attribute Consistency": RELATIONSHIP_ANSWER_TEMPLATE ,
}

category_to_answer_prompt_stage_1 = {
    "Appearance Quality": REF_FREE_APPEARANCE_ANSWER_TEMPLATE_STAGE_1,
    "Appearance Quality + Ref": REF_BASED_APPEARANCE_ANSWER_TEMPLATE_STAGE_1,
}

category_to_answer_prompt_stage_2 = {
    "Appearance Quality": REF_FREE_APPEARANCE_ANSWER_TEMPLATE_STAGE_2,
    "Appearance Quality + Ref": REF_BASED_APPEARANCE_ANSWER_TEMPLATE_STAGE_2,
}

category_to_eval_prompt = {
    "Intrinsic Attribute Consistency": INTRINSIC_EVAL_TEMPLATE,
    "Relationship Attribute Consistency": RELATIONSHIP_EVAL_TEMPLATE,
}

category_to_eval_prompt_stage_1 = {
    "Intrinsic Attribute Consistency": INTRINSIC_EVAL_TEMPLATE_STAGE_1,
    "Relationship Attribute Consistency": RELATIONSHIP_EVAL_TEMPLATE_STAGE_1,
}

category_to_eval_prompt_stage_2 = {
    "Intrinsic Attribute Consistency": INTRINSIC_EVAL_TEMPLATE_STAGE_2,
    "Relationship Attribute Consistency": RELATIONSHIP_EVAL_TEMPLATE_STAGE_2,
}

category_to_summary_prompt = {
    "Appearance Quality": APPEARANCE_SUMMARIZE_TEMPLATE,
    "Intrinsic Attribute Consistency": INTRINSIC_SUMMARIZE_TEMPLATE,
    "Relationship Attribute Consistency": RELATIONSHIP_SUMMARIZE_TEMPLATE,
}

category_to_summary_prompt_stage_1 = {
    "Appearance Quality": APPEARANCE_SUMMARIZE_TEMPLATE_STAGE_1,
    "Intrinsic Attribute Consistency": INTRINSIC_SUMMARIZE_TEMPLATE_STAGE_1,
    "Relationship Attribute Consistency": RELATIONSHIP_SUMMARIZE_TEMPLATE_STAGE_1,
}

category_to_summary_prompt_stage_2 = {
    "Appearance Quality": APPEARANCE_SUMMARIZE_TEMPLATE_STAGE_2,
    "Intrinsic Attribute Consistency": INTRINSIC_SUMMARIZE_TEMPLATE_STAGE_2,
    "Relationship Attribute Consistency": RELATIONSHIP_SUMMARIZE_TEMPLATE_STAGE_2,
}

category_long_to_short = {
    "Appearance Quality": "appearance",
    "Intrinsic Attribute Consistency": "intrinsic",
    "Relationship Attribute Consistency": "relationship",
}


def dump_data(data_dict: dict, output_dir: str, readable: bool = False):
    assert isinstance(data_dict, dict)
    for _, value in data_dict.items():
        assert isinstance(value, list)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    for key, value in data_dict.items():
        if readable:
            _output_dir = os.path.join(output_dir, key)
            if not os.path.exists(_output_dir):
                os.makedirs(_output_dir, exist_ok=True)
            for sample in value:
                with open(
                    os.path.join(_output_dir, f"{sample['id']}.md"), "w+", encoding="utf-8"
                ) as f:
                    for _key, _value in sample.items():
                        f.write(f"# {_key}\n\n---\n\n")
                        if _key == "history":
                            for item in _value:
                                f.write(f"{item[0]}\n\n---\n\n")
                                f.write(f"{item[1]}\n\n---\n\n")
                        else:
                            f.write(f"{_value}\n\n---\n\n")
        else:
            with open(
                os.path.join(output_dir, f"{key}.json"), "w+", encoding="utf-8"
            ) as f:
                json.dump(obj=value, fp=f, indent=4, ensure_ascii=False)


class T2VEvalSFTDataConstructor:
    def __init__(self, data_file: str, image_dir: str = None, disable_ref: bool = False):
        """Sample format: {
            "id": int,
            "model_name": "sd15" / "sd3" / "sdxl",
            "gt_image": "xxx.jpg",
            "ref_image": "xxx.jpg" / null,
            "image_caption": "<caption>",
            "data": {
                "Structure Information": {
                    "Intrinsic Attributes": {
                        "<Entity 1>": [
                            "attribute 1: xxx",
                            "attribute 2: xxx",
                            ...
                        ],
                        "<Entity 2>": [
                            ...
                        ],
                        ...
                    },
                    "Relationship Attributes": {
                        "<Relation 1>": [
                            "entities involved: <entity 1>, <entity 2>, ...",
                            "value: <Relation>"
                        ],
                        ...
                    }
                },
                "Questions": {
                    "Appearance Quality Questions": {
                        "<Entity 1>": [
                            {
                                "question": "<question>",
                                "value": null
                            }
                        ],
                        ...
                    },
                    "Intrinsic Attribute Consistency Questions": {
                        "<Entity 1>": [
                            {
                                "question": "<question>",
                                "value": null
                            }
                        ],
                        ...
                    },
                    "Relationship Attribute Consistency Questions": [
                        {
                            "question": "<question>",
                            "value": {
                                "entities": "<entity 1>, <entity 2>"
                            }
                        },
                        ...
                    ]
                },
                "Image Caption": {
                    "<Entity 1>": [
                        "caption: <caption>"
                    ],
                    "<Entity 2>": [
                        "caption: <caption>"
                    ],
                    ...
                },
                "Answers": {
                    "Appearance Quality Questions": {
                        "<Entity 1>": [
                            {
                                "question": "<question>",
                                "value": {
                                    "explanation": "<explanation>",
                                    "score": float
                                }
                            }
                        ],
                        ...
                    },
                    "Intrinsic Attribute Consistency Questions": {
                        "<Entity 1>": [
                            {
                                "question": "<question>",
                                "value": {
                                    "answer": "<answer>"
                                }
                            },
                            ...
                        ],
                        ...
                    },
                    "Relationship Attribute Consistency Questions": [
                        {
                            "question": "<question>",
                            "value": {
                                "entities": "<entity 1>, ...",
                                "answer": "<answer>"
                            }
                        },
                        ...
                    ]
                },
                "Evaluation": {
                    "Appearance Quality Answers": {
                        "<Entity 1>": [
                            {
                                "question": "<question>",
                                "value": {
                                    "explanation": "<explanation>",
                                    "score": float
                                }
                            }
                        ],
                        ...
                    },
                    "Intrinsic Attribute Consistency Answers": {
                        "<Entity 1>": [
                            {
                                "question": "<question>",
                                "value": {
                                    "answer": "<answer>",
                                    "explanation": "<explanation>",
                                    "score": float
                                }
                            },
                            ...
                        ],
                        ...
                    },
                    "Relationship Attribute Consistency Answers": [
                        {
                            "question": "<question>",
                            "value": {
                                "entities": "<entity 1>, <entity 2>, ...",
                                "answer": "<answer>",
                                "explanation": "<exlanation>",
                                "score": float
                            }
                        }
                    ],
                    "Overall Evaluation": {
                        "Appearance Quality Summary": {
                            "explanation": "<explanation>",
                            "score": float
                        },
                        "Intrinsic Attribute Consistency Summary": {
                            "explanation": "<explanation>",
                            "score": float
                        },
                        "Relationship Attribute Consistency Summary": {
                            "explanation": "<explanation>",
                            "score": float
                        },
                        "Overall Score": {
                            "explanation": "<explanation>",
                            "score": float
                        }
                    }
                }
            }
        }


        Args:
            data_file (str): _description_
            output_dir (str): _description_
        """
        self.image_dir = image_dir
        self.disable_ref = disable_ref
        self.orig_image_placeholder = "<ImagePlaceholder>"
        with open(data_file, "r+", encoding="utf-8") as f:
            self.data = json.load(f)

    @abstractmethod
    def replace_image_placeholder(self, text: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def conv_template(
        self,
        gt_image: str,
        query: str,
        response: str,
        history: list = [],
        ref_image: str = None,
        id: str = None,
    ) -> dict:
        raise NotImplementedError

    def _get_extract_structure_info_by_index(self, sample_index):
        return {
            "Structure Information": self.data[sample_index]["data"][
                "Structure Information"
            ],
            "Questions": self.data[sample_index]["data"]["Questions"],
            "Image Caption": self.data[sample_index]["data"]["Image Caption"],
        }

    def construct_all(self, include_multi_stage: bool = False, separate_aspects: bool = False, include_all_in_one: bool = False, add_ablation_1: bool = False):
        all_data = dict()
        for index in tqdm(range(len(self.data))):
            single_data = self._construct_sample_single(
                sample_index=index, include_multi_stage=include_multi_stage, separate_aspects=separate_aspects, include_all_in_one=include_all_in_one, add_ablation_1=add_ablation_1
            )
            for key, value in single_data.items():
                if key in all_data:
                    if isinstance(value, dict):
                        all_data[key].append(value)
                    elif isinstance(value, list):
                        all_data[key].extend(value)
                else:
                    if isinstance(value, dict):
                        all_data[key] = [value]
                    elif isinstance(value, list):
                        all_data[key] = value
        return all_data

    def _construct_sample_single(
        self, sample_index: int, include_multi_stage: bool = False, separate_aspects: bool = False, include_all_in_one: bool = False, add_ablation_1: bool = False
    ):
        sample = {
            "extract": self._construct_extract_sample_single(sample_index=sample_index),
            "appearance": self._construct_appearance_answer_sample_single(
                sample_index=sample_index
            ),
            "intrinsic": self._construct_intrinsic_answer_and_eval_sample_single(
                sample_index=sample_index
            ),
            "relationship": self._construct_relationship_answer_and_eval_sample_single(
                sample_index=sample_index
            ),
            "summarize" if separate_aspects else "summarize-all-in-one": self._construct_summarize_sample_single(
                sample_index=sample_index,
                separate_aspects=separate_aspects
            ),
        }
        if include_multi_stage:
            appearance_multi = self._construct_appearance_answer_sample_single(sample_index=sample_index, multi_stage=True)
            intrinsic_multi = self._construct_intrinsic_answer_and_eval_sample_single(sample_index=sample_index, multi_stage=True)
            relationship_multi = self._construct_relationship_answer_and_eval_sample_single(sample_index=sample_index, multi_stage=True)
            summarize_multi = self._construct_summarize_sample_single(sample_index=sample_index, multi_stage=True, separate_aspects=separate_aspects)
            sample.update(
                {
                    "appearance-multi-stage_1": [sample for sample in appearance_multi if sample['id'].endswith('stage_1')],
                    "appearance-multi-stage_2": [sample for sample in appearance_multi if sample['id'].endswith('stage_2')],
                    "intrinsic-multi-stage_1": [sample for sample in intrinsic_multi if sample['id'].endswith('stage_1')],
                    "intrinsic-multi-stage_2": [sample for sample in intrinsic_multi if sample['id'].endswith('stage_2')],
                    "relationship-multi-stage_1": [sample for sample in relationship_multi if sample['id'].endswith('stage_1')],
                    "relationship-multi-stage_2": [sample for sample in relationship_multi if sample['id'].endswith('stage_2')],
                    "summarize-multi-stage_1": [sample for sample in summarize_multi if sample['id'].endswith('stage_1')],
                    "summarize-multi-stage_2": [sample for sample in summarize_multi if sample['id'].endswith('stage_2')],
                }
            )
        if include_all_in_one:
            appearance_all_in_one = self._construct_appearance_answer_sample_single(sample_index=sample_index, all_in_one=True)
            intrinsic_all_in_one = self._construct_intrinsic_answer_and_eval_sample_single(sample_index=sample_index, all_in_one=True)
            relationship_all_in_one = self._construct_relationship_answer_and_eval_sample_single(sample_index=sample_index, all_in_one=True)
            sample.update(
                {
                    "appearance-qa-all-in-one": appearance_all_in_one,
                    "intrinsic-qa-all-in-one": intrinsic_all_in_one,
                    "relationship-qa-all-in-one": relationship_all_in_one
                }
            )
        if add_ablation_1:
            ablation_1 = self._construct_answer_and_eval_sample_single_ablation_1(sample_index=sample_index)
            sample.update(
                {
                    "qa-all-in-one": ablation_1
                }
            )
        return sample

    def _construct_extract_sample_single(self, sample_index: int, multi_stage: bool = False):
        if not multi_stage:
            return self.conv_template(
                gt_image=self.data[sample_index]["gt_image"],
                query=self.replace_image_placeholder(
                    EXTRACT_TEMPLATE.format(
                        text_prompt=self.data[sample_index]["image_caption"]
                    )
                ),
                response=json_to_markdown(
                    self._get_extract_structure_info_by_index(sample_index=sample_index)
                ),
                history=[],
                id=str(sample_index),
            )
        else:
            pass
        
    def _construct_answer_and_eval_sample_single_ablation_1(self, sample_index: int):
        all_in_one_sample = {
            'gt_image': None,
            'question': {
                'Appearance Quality Questions': {},
                'Intrinsic Attribute Consistency Questions': {},
                'Relationship Attribute Consistency Questions': []
            },
            'answer': {
                'Appearance Quality Questions': {},
                'Intrinsic Attribute Consistency Questions': {},
                'Relationship Attribute Consistency Questions': []
            },
            'evaluation': {
                'Appearance Quality Answers': {},
                'Intrinsic Attribute Consistency Answers': {},
                'Relationship Attribute Consistency Answers': []
            },
            'history': [],
            'ref_image': None,
            'id': f"{sample_index}-qa-all-in-one"
        }
        category = "Appearance Quality"
        for entity_id, entity in enumerate(list(
            self.data[sample_index]["data"]["Structure Information"][
                "Intrinsic Attributes"
            ].keys()
        )):
            for i, (question, answer) in enumerate(
                zip(
                    self.data[sample_index]["data"]["Questions"][f"{category} Questions"][entity],
                    self.data[sample_index]["data"]["Answers"][f"{category} Questions"][entity]
                )
            ):
                all_in_one_sample['gt_image'] = self.data[sample_index]["gt_image"]
                if entity not in all_in_one_sample['question'][f"{category} Questions"]:
                    all_in_one_sample['question'][f"{category} Questions"][entity] = [question]
                else:
                    all_in_one_sample['question'][f"{category} Questions"][entity] += [question]
                if entity not in all_in_one_sample['answer'][f"{category} Questions"]:
                    all_in_one_sample['answer'][f"{category} Questions"][entity] = [answer]
                else:
                    all_in_one_sample['answer'][f"{category} Questions"][entity] += [answer]
                if entity not in all_in_one_sample['evaluation'][f"{category} Answers"]:
                    all_in_one_sample['evaluation'][f"{category} Answers"][entity] = [answer]
                else:
                    all_in_one_sample['evaluation'][f"{category} Answers"][entity] += [answer]
                all_in_one_sample['ref_image'] = self.data[sample_index]["ref_image"] if not self.disable_ref else None
                    
        category = "Intrinsic Attribute Consistency"
        for entity_id, entity in enumerate(list(
            self.data[sample_index]["data"]["Structure Information"][
                "Intrinsic Attributes"
            ].keys()
        )):
            for i, (question, answer, evaluation) in enumerate(
                zip(
                    self.data[sample_index]["data"]["Questions"][f"{category} Questions"][entity],
                    self.data[sample_index]["data"]["Answers"][f"{category} Questions"][entity],
                    self.data[sample_index]["data"]["Evaluation"][f"{category} Answers"][entity],
                )
            ):
                all_in_one_sample['gt_image'] = self.data[sample_index]["gt_image"]
                if entity not in all_in_one_sample['question'][f"{category} Questions"]:
                    all_in_one_sample['question'][f"{category} Questions"][entity] = [question]
                else:
                    all_in_one_sample['question'][f"{category} Questions"][entity] += [question]
                if entity not in all_in_one_sample['answer'][f"{category} Questions"]:
                    all_in_one_sample['answer'][f"{category} Questions"][entity] = [answer]
                else:
                    all_in_one_sample['answer'][f"{category} Questions"][entity] += [answer]
                if entity not in all_in_one_sample['evaluation'][f"{category} Answers"]:
                    all_in_one_sample['evaluation'][f"{category} Answers"][entity] = [evaluation]
                else:
                    all_in_one_sample['evaluation'][f"{category} Answers"][entity] += [evaluation]
                    
        category = "Relationship Attribute Consistency"
        for i, (question, answer, evaluation) in enumerate(
            zip(
                self.data[sample_index]["data"]["Questions"][f"{category} Questions"],
                self.data[sample_index]["data"]["Answers"][f"{category} Questions"],
                self.data[sample_index]["data"]["Evaluation"][f"{category} Answers"],
            )
        ):
            all_in_one_sample['gt_image'] = self.data[sample_index]["gt_image"]
            all_in_one_sample['question'][f"{category} Questions"] += [question]
            all_in_one_sample['answer'][f"{category} Questions"] += [answer]
            all_in_one_sample['evaluation'][f"{category} Answers"] += [evaluation]
        
        sample_list = [
            self.conv_template(
                gt_image=all_in_one_sample["gt_image"],
                query=self.replace_image_placeholder(
                    (
                        REF_BASED_ANSWER_TEMPLATE_ABLATION_1 
                        if all_in_one_sample['ref_image'] is not None 
                        else REF_FREE_ANSWER_TEMPLATE_ABLATION_1
                    ).format(questions=json_to_markdown(all_in_one_sample['question'], title_level=1, ignore_score=True))
                ),
                response=json_to_markdown({'Answer': all_in_one_sample['answer']}),
                history=[],
                ref_image=all_in_one_sample['ref_image'],
                id=all_in_one_sample['id'] + '-answer',
            ),
            self.conv_template(
                gt_image=None,
                query=EVAL_TEMPLATE_ABLATION_1.format(
                    structure_info=json_to_markdown(
                        self._get_extract_structure_info_by_index(
                            sample_index=sample_index
                        )
                    ),
                    answers=json_to_markdown(
                        all_in_one_sample['answer'],
                        title_level=1
                    )
                ),
                response=json_to_markdown({'Evaluation': all_in_one_sample['evaluation']}),
                history=[],
                ref_image=None,
                id=all_in_one_sample['id'] + '-eval'
            )
        ]
        return sample_list

    def _construct_appearance_answer_sample_single(
        self, sample_index: int, multi_stage: bool = False, all_in_one: bool = False
    ):
        sample_list = []
        all_in_one_sample = {
            'gt_image': None,
            'query': {},
            'response': {'Answer': {}},
            'history': [],
            'ref_image': None,
            'id': f"{sample_index}-appearance-qa-all-in-one"
        }
        category = "Appearance Quality"
        for entity_id, entity in enumerate(list(
            self.data[sample_index]["data"]["Structure Information"][
                "Intrinsic Attributes"
            ].keys()
        )):
            for i, (question, answer) in enumerate(
                zip(
                    self.data[sample_index]["data"]["Questions"][f"{category} Questions"][entity],
                    self.data[sample_index]["data"]["Answers"][f"{category} Questions"][entity],
                )
            ):
                if (
                    self.data[sample_index]["ref_image"] is not None
                    and not self.disable_ref
                ):
                    _category = category + " + Ref"
                else:
                    _category = category
                if not all_in_one:
                    sample_list.append(
                        self.conv_template(
                            gt_image=self.data[sample_index]["gt_image"],
                            query=self.replace_image_placeholder(
                                (
                                    category_to_answer_prompt
                                    if not multi_stage
                                    else category_to_answer_prompt_stage_1
                                )[_category].format(question=json_to_markdown(question))
                            ),
                            response=json_to_markdown(
                                {"Answer": {entity: answer}}, ignore_score=multi_stage
                            ),
                            history=[],
                            ref_image=(
                                self.data[sample_index]["ref_image"]
                                if not self.disable_ref
                                else None
                            ),
                            id=(
                                f"{sample_index}-{entity_id}-{i}"
                                if not multi_stage
                                else f"{sample_index}-{entity_id}-{i}-stage_1"
                            ),
                        )
                    )
                else:
                    all_in_one_sample['gt_image'] = self.data[sample_index]["gt_image"]
                    if entity not in all_in_one_sample['query']:
                        all_in_one_sample['query'][entity] = [question]
                    else:
                        all_in_one_sample['query'][entity] += [question]
                    if entity not in all_in_one_sample['response']['Answer']:
                        all_in_one_sample['response']['Answer'][entity] = [answer]
                    else:
                        all_in_one_sample['response']['Answer'][entity] += [answer]
                    all_in_one_sample['ref_image'] = self.data[sample_index]["ref_image"] if not self.disable_ref else None
                if multi_stage:
                    sample_list.append(
                        self.conv_template(
                            gt_image=self.data[sample_index]["gt_image"],
                            query=self.replace_image_placeholder(
                                category_to_answer_prompt_stage_2[_category].format(
                                    question_and_exp=json_to_markdown(
                                        answer, ignore_score=True
                                    )
                                )
                            ),
                            response=(
                                str(answer["value"]["score"])
                                if answer["value"] is not None
                                and "score" in answer["value"]
                                else "N/A"
                            ),
                            history=[],
                            ref_image=(
                                self.data[sample_index]["ref_image"]
                                if not self.disable_ref
                                else None
                            ),
                            id=f"{sample_index}-{entity_id}-{i}-stage_2",
                        )
                    )
        if all_in_one:
            sample_list = [
                self.conv_template(
                    gt_image=all_in_one_sample["gt_image"],
                    query=self.replace_image_placeholder(
                        (
                            REF_BASED_APPEARANCE_ANSWER_TEMPLATE_ABLATION_2 
                            if all_in_one_sample['ref_image'] is not None 
                            else REF_FREE_APPEARANCE_ANSWER_TEMPLATE_ABLATION_2
                        ).format(questions=json_to_markdown(all_in_one_sample['query'], title_level=1))
                    ),
                    response=json_to_markdown(all_in_one_sample['response']),
                    history=[],
                    ref_image=all_in_one_sample['ref_image'],
                    id=all_in_one_sample['id'],
                )
            ]
        return sample_list

    def _construct_intrinsic_answer_and_eval_sample_single(
        self, sample_index: int, multi_stage: bool = False, all_in_one: bool = False
    ):
        sample_list = []
        all_in_one_sample = {
            'gt_image': None,
            'question': {},
            'answer': {'Answer': {}},
            'evaluation': {'Evaluation': {}},
            'history': [],
            'ref_image': None,
            'id': f"{sample_index}-intrinsic-qa-all-in-one"
        }
        category = "Intrinsic Attribute Consistency"
        for entity_id, entity in enumerate(list(
            self.data[sample_index]["data"]["Structure Information"][
                "Intrinsic Attributes"
            ].keys()
        )):
            for i, (question, answer, evaluation) in enumerate(
                zip(
                    self.data[sample_index]["data"]["Questions"][f"{category} Questions"][entity],
                    self.data[sample_index]["data"]["Answers"][f"{category} Questions"][entity],
                    self.data[sample_index]["data"]["Evaluation"][f"{category} Answers"][entity],
                )
            ):
                if not all_in_one:
                    sample_list.append(
                        self.conv_template(
                            gt_image=self.data[sample_index]["gt_image"],
                            query=(
                                INTRINSIC_EVAL_TEMPLATE
                                if not multi_stage
                                else INTRINSIC_EVAL_TEMPLATE_STAGE_1
                            ).format(
                                answer=json_to_markdown({"Answer": {entity: answer}}),
                                structure_info=json_to_markdown(
                                    self._get_extract_structure_info_by_index(
                                        sample_index=sample_index
                                    )
                                ),
                            ),
                            response=json_to_markdown(
                                {"Evaluation": {entity: evaluation}},
                                ignore_score=multi_stage,
                            ),
                            history=[
                                [
                                    self.replace_image_placeholder(
                                        INTRINSIC_ANSWER_TEMPLATE.format(
                                            question=json_to_markdown(question)
                                        )
                                    ),
                                    json_to_markdown({"Answer": {entity: answer}}),
                                ]
                            ],
                            id=(
                                f"{sample_index}-{entity_id}-{i}"
                                if not multi_stage
                                else f"{sample_index}-{entity_id}-{i}-stage_1"
                            ),
                        )
                    )
                else:
                    all_in_one_sample['gt_image'] = self.data[sample_index]["gt_image"]
                    if entity not in all_in_one_sample['question']:
                        all_in_one_sample['question'][entity] = [question]
                    else:
                        all_in_one_sample['question'][entity] += [question]
                    if entity not in all_in_one_sample['answer']['Answer']:
                        all_in_one_sample['answer']['Answer'][entity] = [answer]
                    else:
                        all_in_one_sample['answer']['Answer'][entity] += [answer]
                    if entity not in all_in_one_sample['evaluation']['Evaluation']:
                        all_in_one_sample['evaluation']['Evaluation'][entity] = [evaluation]
                    else:
                        all_in_one_sample['evaluation']['Evaluation'][entity] += [evaluation]
                if multi_stage:
                    sample_list.append(
                        self.conv_template(
                            gt_image=self.data[sample_index]["gt_image"],
                            query=self.replace_image_placeholder(
                                INTRINSIC_EVAL_TEMPLATE_STAGE_2.format(
                                    answer_and_exp=json_to_markdown(
                                        {"Evaluation": {entity: evaluation}},
                                        ignore_score=True
                                    ),
                                    structure_info=json_to_markdown(
                                        self._get_extract_structure_info_by_index(
                                            sample_index=sample_index
                                        )
                                    )
                                ),
                            ),
                            response=(
                                str(evaluation["value"]["score"])
                                if evaluation["value"] is not None
                                and "score" in evaluation["value"]
                                else "N/A"
                            ),
                            history=[],
                            id=f"{sample_index}-{entity_id}-{i}-stage_2",
                        )
                    )
        if all_in_one:
            sample_list = [
                self.conv_template(
                    gt_image=all_in_one_sample["gt_image"],
                    query=self.replace_image_placeholder(
                        INTRINSIC_ANSWER_TEMPLATE_ABLATION_2.format(questions=json_to_markdown(all_in_one_sample['question'], title_level=1))
                    ),
                    response=json_to_markdown(all_in_one_sample['answer']),
                    history=[],
                    ref_image=None,
                    id=all_in_one_sample['id'] + '-answer',
                ),
                self.conv_template(
                    gt_image=None,
                    query=INTRINSIC_EVAL_TEMPLATE_ABLATION_2.format(
                        structure_info=json_to_markdown(
                            self._get_extract_structure_info_by_index(
                                sample_index=sample_index
                            )
                        ),
                        answers=json_to_markdown(
                            all_in_one_sample['answer']['Answer'],
                            title_level=1
                        )
                    ),
                    response=json_to_markdown(all_in_one_sample['evaluation']),
                    history=[],
                    ref_image=None,
                    id=all_in_one_sample['id'] + '-eval'
                )
            ]
        return sample_list

    def _construct_relationship_answer_and_eval_sample_single(
        self, sample_index: int, multi_stage: bool = False, all_in_one: bool = False
    ):
        sample_list = []
        category = "Relationship Attribute Consistency"
        all_in_one_sample = {
            'gt_image': None,
            'question': [],
            'answer': {'Answer': []},
            'evaluation': {'Evaluation': []},
            'history': [],
            'ref_image': None,
            'id': f"{sample_index}-relationship-qa-all-in-one"
        }
        for i, (question, answer, evaluation) in enumerate(
            zip(
                self.data[sample_index]["data"]["Questions"][f"{category} Questions"],
                self.data[sample_index]["data"]["Answers"][f"{category} Questions"],
                self.data[sample_index]["data"]["Evaluation"][f"{category} Answers"],
            )
        ):
            if not all_in_one:
                sample_list.append(
                    self.conv_template(
                        gt_image=self.data[sample_index]["gt_image"],
                        query=(
                            RELATIONSHIP_EVAL_TEMPLATE
                            if not multi_stage
                            else RELATIONSHIP_EVAL_TEMPLATE_STAGE_1
                        ).format(
                            answer=json_to_markdown({"Answer": answer}),
                            structure_info=json_to_markdown(
                                self._get_extract_structure_info_by_index(
                                    sample_index=sample_index
                                )
                            ),
                        ),
                        response=json_to_markdown(
                            {"Evaluation": evaluation}, ignore_score=multi_stage
                        ),
                        history=[
                            [
                                self.replace_image_placeholder(
                                    RELATIONSHIP_ANSWER_TEMPLATE .format(
                                        question=json_to_markdown(question)
                                    )
                                ),
                                json_to_markdown({"Answer": answer}),
                            ]
                        ],
                        id=(
                            f"{sample_index}-{i}"
                            if not multi_stage
                            else f"{sample_index}-{i}-stage_1"
                        ),
                    )
                )
            else:
                all_in_one_sample['gt_image'] = self.data[sample_index]["gt_image"]
                all_in_one_sample['question'] += [question]
                all_in_one_sample['answer']['Answer'] += [answer]
                all_in_one_sample['evaluation']['Evaluation'] += [evaluation]
            if multi_stage:
                sample_list.append(
                    self.conv_template(
                        gt_image=self.data[sample_index]["gt_image"],
                        query=self.replace_image_placeholder(
                            RELATIONSHIP_EVAL_TEMPLATE_STAGE_2.format(
                                answer_and_exp=json_to_markdown(
                                    {"Evaluation": evaluation},
                                    ignore_score=True
                                ),
                                structure_info=json_to_markdown(
                                    self._get_extract_structure_info_by_index(
                                        sample_index=sample_index
                                    )
                                )
                            )
                        ),
                        response=(
                            str(evaluation["value"]["score"])
                            if evaluation["value"] is not None
                            and "score" in evaluation["value"]
                            else "N/A"
                        ),
                        history=[],
                        id=f"{sample_index}-{i}-stage_2",
                    )
                )
        if all_in_one:
            sample_list = [
                self.conv_template(
                    gt_image=all_in_one_sample["gt_image"],
                    query=self.replace_image_placeholder(
                        RELATIONSHIP_ANSWER_TEMPLATE_ABLATION_2.format(questions=json_to_markdown(all_in_one_sample['question'], title_level=1))
                    ),
                    response=json_to_markdown(all_in_one_sample['answer']),
                    history=[],
                    ref_image=None,
                    id=all_in_one_sample['id'] + '-answer',
                ),
                self.conv_template(
                    gt_image=None,
                    query=RELATIONSHIP_EVAL_TEMPLATE_ABLATION_2.format(
                        structure_info=json_to_markdown(
                            self._get_extract_structure_info_by_index(
                                sample_index=sample_index
                            )
                        ),
                        answers=json_to_markdown(
                            all_in_one_sample['answer']['Answer'],
                            title_level=1
                        )
                    ),
                    response=json_to_markdown(all_in_one_sample['evaluation']),
                    history=[],
                    ref_image=None,
                    id=all_in_one_sample['id'] + '-eval'
                )
            ]
        return sample_list

    def _construct_summarize_sample_single(
        self, sample_index: int, multi_stage: bool = False, separate_aspects: bool = False
    ):
        evaluation_input = {
            "Appearance Quality Answers": self.data[sample_index]["data"]["Evaluation"][
                "Appearance Quality Answers"
            ],
            "Intrinsic Attribute Consistency Answers": self.data[sample_index]["data"][
                "Evaluation"
            ]["Intrinsic Attribute Consistency Answers"],
            "Relationship Attribute Consistency Answers": self.data[sample_index][
                "data"
            ]["Evaluation"]["Relationship Attribute Consistency Answers"],
        }
        if not multi_stage:
            if not separate_aspects:
                return self.conv_template(
                    gt_image=self.data[sample_index]["gt_image"],
                    query=self.replace_image_placeholder(
                        OVERALL_SUMMARIZE_TEMPLATE.format(
                            eval_result=json_to_markdown(evaluation_input),
                            structure_info=json_to_markdown(
                                self._get_extract_structure_info_by_index(
                                    sample_index=sample_index
                                )
                            ),
                        )
                    ),
                    response=json_to_markdown(
                        self.data[sample_index]["data"]["Evaluation"]["Overall Evaluation"],
                        is_overall_eval=True,
                    ),
                    history=[],
                    id=str(sample_index),
                )
            else:
                return [
                    self.conv_template(
                        gt_image=self.data[sample_index]["gt_image"],
                        query=self.replace_image_placeholder(
                            category_to_summary_prompt[category].format(
                                eval_result=json_to_markdown({f"{category} Answers": evaluation_input[f"{category} Answers"]}),
                                structure_info=json_to_markdown(
                                    self._get_extract_structure_info_by_index(
                                        sample_index=sample_index
                                    )
                                ),
                            )
                        ),
                        response=json_to_markdown(
                            {f"{category} Summary": self.data[sample_index]["data"]["Evaluation"]["Overall Evaluation"][f"{category} Summary"]}
                            if f"{category} Summary" in self.data[sample_index]["data"]["Evaluation"]["Overall Evaluation"] else self.data[sample_index]["data"]["Evaluation"]["Overall Evaluation"],
                            is_overall_eval=True,
                        ),
                        history=[],
                        id=f"{sample_index}-{category_long_to_short[category]}",
                    ) for i, category in enumerate(["Appearance Quality", "Intrinsic Attribute Consistency", "Relationship Attribute Consistency"])
                ] + [
                    self.conv_template(
                        gt_image=self.data[sample_index]["gt_image"],
                        query=self.replace_image_placeholder(
                            MERGE_SUMMARIZE_TEMPLATE.format(
                                eval_result=json_to_markdown(evaluation_input) + json_to_markdown(
                                    {key: value for key, value in self.data[sample_index]["data"]["Evaluation"]["Overall Evaluation"].items() if key != "Overall Score"},
                                    is_overall_eval=True,
                                ),
                                structure_info=json_to_markdown(
                                    self._get_extract_structure_info_by_index(
                                        sample_index=sample_index
                                    )
                                ),
                            )
                        ),
                        response=json_to_markdown(
                            {f"Overall Score": self.data[sample_index]["data"]["Evaluation"]["Overall Evaluation"]["Overall Score"]},
                            is_overall_eval=True,
                        ),
                        history=[],
                        id=f"{sample_index}-overall",
                    )
                ]
        else:
            temp_dict = self.data[sample_index]["data"]["Evaluation"][
                "Overall Evaluation"
            ]
            if isinstance(temp_dict, dict):
                appearance_score = (
                    str(temp_dict["Appearance Quality Summary"]["score"])
                    if isinstance(temp_dict["Appearance Quality Summary"], dict)
                    and "score" in temp_dict["Appearance Quality Summary"]
                    else "N/A"
                )
                intrinsic_score = (
                    str(temp_dict["Intrinsic Attribute Consistency Summary"]["score"])
                    if isinstance(
                        temp_dict["Intrinsic Attribute Consistency Summary"], dict
                    )
                    and "score" in temp_dict["Intrinsic Attribute Consistency Summary"]
                    else "N/A"
                )
                relationship_score = (
                    str(temp_dict["Relationship Attribute Consistency Summary"]["score"])
                    if isinstance(
                        temp_dict["Relationship Attribute Consistency Summary"], dict
                    )
                    and "score"
                    in temp_dict["Relationship Attribute Consistency Summary"]
                    else "N/A"
                )
                overall_score = (
                    str(temp_dict["Overall Score"]["score"])
                    if isinstance(temp_dict["Overall Score"], dict)
                    and "score" in temp_dict["Overall Score"]
                    else "N/A"
                )
            else:
                appearance_score = intrinsic_score = relationship_score = (
                    overall_score
                ) = "N/A"
            if not separate_aspects:
                return [
                    self.conv_template(
                        gt_image=self.data[sample_index]["gt_image"],
                        query=self.replace_image_placeholder(
                            OVERALL_SUMMARIZE_TEMPLATE_STAGE_1.format(
                                eval_result=json_to_markdown(evaluation_input),
                                structure_info=json_to_markdown(
                                    self._get_extract_structure_info_by_index(
                                        sample_index=sample_index
                                    )
                                ),
                            )
                        ),
                        response=json_to_markdown(
                            self.data[sample_index]["data"]["Evaluation"][
                                "Overall Evaluation"
                            ],
                            is_overall_eval=True,
                            ignore_score=True,
                        ),
                        history=[],
                        id=f"{sample_index}-stage_1",
                    ),
                    self.conv_template(
                        gt_image=self.data[sample_index]["gt_image"],
                        query=self.replace_image_placeholder(
                            OVERALL_SUMMARIZE_TEMPLATE_STAGE_2.format(
                                eval_result_and_exp=json_to_markdown(evaluation_input)
                                + "\n# Overall Evaluation\n"
                                + json_to_markdown(
                                    self.data[sample_index]["data"]["Evaluation"][
                                        "Overall Evaluation"
                                    ],
                                    is_overall_eval=True,
                                    ignore_score=True,
                                ),
                                structure_info=json_to_markdown(
                                    self._get_extract_structure_info_by_index(
                                        sample_index=sample_index
                                    )
                                ),
                            )
                        ),
                        response=f"{appearance_score} {intrinsic_score} {relationship_score} {overall_score}",
                        history=[],
                        id=f"{sample_index}-stage_2",
                    ),
                ]
            else:
                scores = [appearance_score, intrinsic_score, relationship_score, overall_score]
                return [
                    self.conv_template(
                        gt_image=self.data[sample_index]["gt_image"],
                        query=self.replace_image_placeholder(
                            category_to_summary_prompt_stage_1[category].format(
                                eval_result=json_to_markdown({f"{category} Answers": evaluation_input[f"{category} Answers"]}),
                                structure_info=json_to_markdown(
                                    self._get_extract_structure_info_by_index(
                                        sample_index=sample_index
                                    )
                                ),
                            )
                        ),
                        response=json_to_markdown(
                            {f"{category} Summary": self.data[sample_index]["data"]["Evaluation"]["Overall Evaluation"][f"{category} Summary"]},
                            is_overall_eval=True,
                            ignore_score=True,
                        ),
                        history=[],
                        id=f"{sample_index}-{category_long_to_short[category]}-stage_1",
                    ) for i, category in enumerate(["Appearance Quality", "Intrinsic Attribute Consistency", "Relationship Attribute Consistency"])
                ] + [
                    self.conv_template(
                        gt_image=self.data[sample_index]["gt_image"],
                        query=self.replace_image_placeholder(
                            category_to_summary_prompt_stage_2[category].format(
                                eval_result_and_exp=json_to_markdown({f"{category} Answers": evaluation_input[f"{category} Answers"]})
                                + "\n# Overall Evaluation\n"
                                + json_to_markdown(
                                    {f"{category} Summary": self.data[sample_index]["data"]["Evaluation"]["Overall Evaluation"][f"{category} Summary"]},
                                    is_overall_eval=True,
                                    ignore_score=True,
                                ),
                                structure_info=json_to_markdown(
                                    self._get_extract_structure_info_by_index(
                                        sample_index=sample_index
                                    )
                                ),
                            )
                        ),
                        response=f"{scores[i]}",
                        history=[],
                        id=f"{sample_index}-{category_long_to_short[category]}-stage_2",
                    )  for i, category in enumerate(["Appearance Quality", "Intrinsic Attribute Consistency", "Relationship Attribute Consistency"])
                ] + [
                    self.conv_template(
                        gt_image=self.data[sample_index]["gt_image"],
                        query=self.replace_image_placeholder(
                            MERGE_SUMMARIZE_TEMPLATE_STAGE_1.format(
                                eval_result=json_to_markdown(evaluation_input)
                                + "\n# Overall Evaluation\n"
                                + json_to_markdown(
                                    {key: value for key, value in self.data[sample_index]["data"]["Evaluation"]["Overall Evaluation"].items() if key != "Overall Score"},
                                    is_overall_eval=True
                                ),
                                structure_info=json_to_markdown(
                                    self._get_extract_structure_info_by_index(
                                        sample_index=sample_index
                                    )
                                ),
                            )
                        ),
                        response=json_to_markdown(
                            {"Overall Score": self.data[sample_index]["data"]["Evaluation"]["Overall Evaluation"]["Overall Score"]},
                            is_overall_eval=True,
                            ignore_score=True,
                        ),
                        history=[],
                        id=f"{sample_index}-overall-stage_1",
                    ),
                    self.conv_template(
                        gt_image=self.data[sample_index]["gt_image"],
                        query=self.replace_image_placeholder(
                            MERGE_SUMMARIZE_TEMPLATE_STAGE_2.format(
                                eval_result_and_exp=json_to_markdown(evaluation_input)
                                + "\n# Overall Evaluation\n"
                                + json_to_markdown(
                                    {key: value for key, value in self.data[sample_index]["data"]["Evaluation"]["Overall Evaluation"].items() if key != "Overall Score"},
                                    is_overall_eval=True
                                )
                                + json_to_markdown(
                                    {"Overall Score": self.data[sample_index]["data"]["Evaluation"]["Overall Evaluation"]["Overall Score"]},
                                    is_overall_eval=True,
                                    ignore_score=True
                                ),
                                structure_info=json_to_markdown(
                                    self._get_extract_structure_info_by_index(
                                        sample_index=sample_index
                                    )
                                ),
                            )
                        ),
                        response=f"{overall_score}",
                        history=[],
                        id=f"{sample_index}-overall-stage_2",
                    ),
                ]
