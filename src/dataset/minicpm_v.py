from src.dataset.sft_dataset_constructor import T2VEvalSFTDataConstructor, dump_data


class MiniCPMSFTDataConstructor(T2VEvalSFTDataConstructor):
    def replace_image_placeholder(self, text: str) -> str:
        text_splits = text.split(self.orig_image_placeholder)
        if len(text_splits) > 2:
            text = ''
            for i, split in enumerate(text_splits):
                text += split
                if i != len(text_splits) - 1:
                    text += f"Image-{i + 1}: <image>"
        else:
            text = '<image>'.join(text_splits)
        return text
    
    def conv_template(self, gt_image: str, query: str, response: str, history: list = [], ref_image: str = None, id: str = None):
        if gt_image is None:
            return {
                "id": id,
                "images": [],
                "query": query,
                "response": response,
                "history": history
            }
        else:
            return {
                "id": id,
                "images": [gt_image, ref_image] if ref_image is not None else [gt_image],
                "query": query,
                "response": response,
                "history": history
            }
