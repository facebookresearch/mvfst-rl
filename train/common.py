import argparse


def add_args(parser):
    parser.add_argument(
        "--mode",
        default="train",
        choices=["train", "test", "test_local", "trace"],
        help="test -> remote test, test_local -> local inference.",
    )
    parser.add_argument(
        "--traced_model",
        default="traced_model.pt",
        help="File to write torchscript traced model to (for training) "
        "or read from (for local testing).",
    )
