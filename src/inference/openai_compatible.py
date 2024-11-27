import base64
from PIL import Image
from io import BytesIO
from openai import OpenAI
from src.inference.inference_engine import InferenceEngine


def convert_image_path_to_base64(image_path: str) -> str:
    if image_path.startswith('file://'):
        image_path = image_path[7:]
        
    with open(image_path, "rb") as f:
        byte_data = f.read()
        image_file = BytesIO(byte_data)
        
    image_format = Image.open(image_file).format
    
    byte_data = image_file.getvalue()
    
    base64_str = base64.b64encode(byte_data).decode('utf-8')
    return f'data:image/{image_format};base64,' + base64_str


class OpenAICompatibleInferenceEngine(InferenceEngine):
    def init_model(self, api_key: str = None, base_url: str = None, model_name: str = None):
        if api_key is None:
            api_key = 'pseudo_api_key'
            
        assert base_url is not None
            
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        
        if model_name is None:
            model_name = self.client.models.list().data[0].id
        self.model_name = model_name
    
    def replace_image_placeholder(self, text: str) -> str:
        text_splits = text.split(self.orig_image_placeholder)
        text = '<ImageHere>'.join(text_splits)
        return text
    
    def chat_single_round(self, prompt: str, gt_image: str = None, ref_image: str = None, history=None, retry: bool = False) -> tuple:
        content = []
        if gt_image is not None and history is None:
            splits = prompt.split('<ImageHere>')
            if len(splits) == 1:
                content.extend([
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': convert_image_path_to_base64(image_path=gt_image)
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
                            'url': convert_image_path_to_base64(image_path=gt_image)
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
                            'url': convert_image_path_to_base64(image_path=gt_image)
                        }
                    },
                    {
                        'type': 'text',
                        'text': splits[1]
                    },
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': convert_image_path_to_base64(image_path=ref_image)
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

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages
        )
        history = messages + [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": response.choices[0].message.content
                    }
                ]
            }
        ]
        
        return response.choices[0].message.content, history
