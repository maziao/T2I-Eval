from PIL import Image
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from src.inference.inference_engine import InferenceEngine


class MiniCPMVOfflineInferenceEngine(InferenceEngine):
    def init_model(self, model_name_or_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = LLM(model=model_name_or_path, trust_remote_code=True, limit_mm_per_prompt={"image": 2}, max_model_len=8192, enforce_eager=True)
        self.image_placeholder = "(<image>./</image>)"
        
        stop_tokens = ['<|im_end|>', '<|endoftext|>']
        stop_token_ids = [self.tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
        self.sampling_params = SamplingParams(
            top_k=1,
            stop_token_ids=stop_token_ids, 
            max_tokens=2048
        )
    
    def replace_image_placeholder(self, text: str) -> str:
        text_splits = text.split(self.orig_image_placeholder)
        text = '<ImageHere>'.join(text_splits)
        return text
    
    def convert_openai_messages_to_minicpm_v_inputs(self, messages: list):
        images = []
        new_messages = []
        for message in messages:
            new_content = ''
            for item in message['content']:
                if item['type'] == 'text':
                    new_content += item['text']
                elif item['type'] == 'image_url':
                    new_content += self.image_placeholder
                    images.append(Image.open(item['image_url']['url']).convert("RGB"))
                else:
                    raise ValueError(f"the type of message item must be text or image_url, but got {item['type']}.")
            
            new_messages.append({
                "role": message['role'],
                "content": new_content
            })
        
        prompt = self.tokenizer.apply_chat_template(
            new_messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = {
            "prompt": prompt,
            "multi_modal_data": {
                "image": images
            }
        }
        
        return inputs
    
    def chat_single_round(self, prompt: str, gt_image: str = None, ref_image: str = None, history=None, retry: bool = False) -> tuple:
        content = []
        if gt_image is not None and history is None:
            splits = prompt.split('<ImageHere>')
            if len(splits) == 1:
                content.extend([
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': gt_image
                        }
                    },
                    {
                        'type': 'text',
                        'text': prompt
                    }
                ])
            elif len(splits) == 2:
                assert gt_image is not None and ref_image is None
                content.extend([
                    {
                        'type': 'text',
                        'text': splits[0]
                    },
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': gt_image
                        }
                    },
                    {
                        'type': 'text',
                        'text': splits[1]
                    }
                ])
            else:
                assert len(splits) == 3 and ref_image is not None
                content.extend([
                    {
                        'type': 'text',
                        'text': splits[0]
                    },
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': gt_image
                        }
                    },
                    {
                        'type': 'text',
                        'text': splits[1]
                    },
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': ref_image
                        }
                    },
                    {
                        'type': 'text',
                        'text': splits[2]
                    }
                ])
        else:
            content.append({
                'type': 'text',
                'text': prompt
            })
        
        if history is None:
            messages = [
                {
                    "role": "user",
                    "content": content
                }
            ]
        else:
            messages = history + [
                {
                    "role": "user",
                    "content": content
                }
            ]

        model_inputs = self.convert_openai_messages_to_minicpm_v_inputs(messages=messages)

        outputs = self.model.generate(model_inputs, sampling_params=self.sampling_params)

        history = messages + [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": outputs[0].outputs[0].text
                    }
                ]
            }
        ]
        
        return outputs[0].outputs[0].text, history
