# src/__main__.py
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(prog="src")
    subparsers = parser.add_subparsers(dest="command")

    # Subcomando train
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", default="configs/default.yaml")

    # Subcomando predict
    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--input",  required=True)
    predict_parser.add_argument("--output", required=True)
    predict_parser.add_argument("--config", default="configs/default.yaml")

    args = parser.parse_args()

    if args.command == "train":
        from src.training import train_pipeline
        train_pipeline(args.config)
    elif args.command == "predict":
        from src.inference import run_inference_cli
        run_inference_cli(args.input, args.output, args.config)
    else:
        parser.print_help()