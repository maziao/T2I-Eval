import os
import argparse
from src.dataset.minicpm_v import MiniCPMSFTDataConstructor
from src.dataset.sft_dataset_constructor import dump_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-files', type=str, nargs='+')
    parser.add_argument('--image-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    args = parser.parse_args()
    
    for file in args.data_files:
        constructor = MiniCPMSFTDataConstructor(data_file=file, image_dir=args.image_dir)
        samples = constructor.construct_all(
            include_multi_stage=True,
            separate_aspects=True
        )
        dump_data(samples, os.path.join(args.output_dir, '.'.join(os.path.basename(file).split('.')[:-1])))
