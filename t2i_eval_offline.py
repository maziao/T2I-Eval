import argparse
from src.inference.minicpm_v_offline import MiniCPMVOfflineInferenceEngine
from src.utils.extract_scores import extract_scores_from_result_dir
from src.utils.calc_correlation import calc_correlation_from_result_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, default='data/test/t2i-eval-bench.json')
    parser.add_argument("--ref-score-file", type=str, default='data/test/scores.json')
    parser.add_argument("--image-root", type=str, required=True)
    parser.add_argument("--model-name-or-path", type=str, default=None)
    parser.add_argument("--max-retry", type=int, default=0)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    engine = MiniCPMVOfflineInferenceEngine(
        data_file=args.input_file,
        image_root=args.image_root,
        output_dir=args.output_dir,
        max_retry=args.max_retry,
        model_init_kwargs=dict(
            model_name_or_path=args.model_name_or_path
        )
    )
    
    engine.inference(
        granularity='coarse',
        multi_stage=True,
        simple_answer_and_eval=True
    )

    extract_scores_from_result_dir(result_dir=args.output_dir)
    
    calc_correlation_from_result_dir(result_dir=args.output_dir, ref_score_file=args.ref_score_file)