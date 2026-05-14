import sys
import argparse


def main():
    parser = argparse.ArgumentParser(prog="src")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Train the pipeline")
    train_parser.add_argument("--config", default="configs/default.yaml")

    predict_parser = subparsers.add_parser("predict", help="Run inference")
    predict_parser.add_argument("--input",  required=True)
    predict_parser.add_argument("--output", required=True)
    predict_parser.add_argument("--config", default="configs/default.yaml")

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Score the trained model against a labeled CSV"
    )
    evaluate_parser.add_argument("--input",  required=True)
    evaluate_parser.add_argument("--config", default="configs/default.yaml")

    args = parser.parse_args()

    if args.command == "train":
        from src.training import main as training_main
        sys.argv = ["src", "--config", args.config]
        training_main()

    elif args.command == "predict":
        from src.inference import main as inference_main
        sys.argv = ["src", "--input", args.input, "--output", args.output, "--config", args.config]
        inference_main()

    elif args.command == "evaluate":
        from src.evaluate_cli import main as evaluate_main
        sys.argv = ["src", "--input", args.input, "--config", args.config]
        evaluate_main()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()