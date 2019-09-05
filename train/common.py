import argparse


def add_args(parser):
    parser.add_argument(
        "--mode",
        default="train",
        choices=["train", "test", "trace"],
        help="Training or test mode.",
    )
