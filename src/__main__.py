# ============================================================================
# Punto de entrada CLI
#
# Autores:
#   Iván Alexander Ramos Ramírez       A01750817
#   Miguel Ángel Galicia Sánchez       A01750744
#   Aislinn Ruiz Sandoval               A01750687
#   Víctor Alejandro Morales García    A01749831
# ============================================================================
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
         # Cuando llamas: python main.py train --config mi.yaml
        from src.training import train_pipeline
        train_pipeline(config_path=args.config)

    elif args.command == "predict":
        from src.inference import run_inference_cli
        run_inference_cli(input_path=args.input,output_path=args.output,config_path=args.config)

    elif args.command == "evaluate":
        from src.evaluate_cli import run_evaluation_cli
        run_evaluation_cli(input_path=args.input,config_path=args.config)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()