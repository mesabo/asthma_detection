import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Asthma Detection Pipeline CLI")
    parser.add_argument(
        "--mode", type=str, default="train",
        choices=["train", "evaluate"],
        help="Operation mode: train or evaluate"
    )
    parser.add_argument(
        "--data", type=str,
        default="./Dataset/Asthma Detection Dataset Version 2/",
        help="Path to the dataset root directory"
    )
    return parser.parse_args()
