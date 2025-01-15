from pipelines.train_pipeline import run_training_pipeline
from pipelines.inference_pipeline import run_inference_pipeline
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'inference'], help='Mode to run the pipeline')
    parser.add_argument('--data_path', type=str, help='Path to dataset file (for training)')
    parser.add_argument('--save_model', type=str, help='Path to save the trained model (for training)')
    parser.add_argument('--model_path', type=str, help='Path to the trained model (for inference)')
    parser.add_argument('--input_texts', type=str, nargs='+', help='Input texts for inference')
    args = parser.parse_args()

    if args.mode == 'train':
        if not args.data_path or not args.save_model:
            raise ValueError('For training, --data_path and --save_model are required.')
        run_training_pipeline(args.data_path, args.save_model)

    elif args.mode == 'inference':
        if not args.model_path or not args.input_texts:
            raise ValueError('For inference, --model_path and --input_texts are required.')
        tokenizer = None  # Load or initialize your tokenizer
        predictions = run_inference_pipeline(args.model_path, tokenizer, args.input_texts)
        for text, sentiment in zip(args.input_texts, predictions):
            print(f"Text: {text} => Sentiment: {sentiment}")

if __name__ == "__main__":
    main()
