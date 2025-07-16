#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 16/07/2025
🚀 Welcome to the Awesome Python Script 🚀

User: messou
Email: mesabo18@gmail.com / messouaboya17@gmail.com
Github: https://github.com/mesabo
Univ: Hosei University, IIST
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

import os
import argparse
import torch

from model.network import ConvLSTMNet
from model.train import train_model
from model.evaluate import evaluate_model
from utils.audio_processing import load_and_process_dataset
from utils.dataset import LungSoundDataset
from utils.config import BEST_MODEL_FILE
from utils.argument_parser import get_args

from torch.utils.data import DataLoader

def main(mode="train", dataset_path="./Dataset/Asthma Detection Dataset Version 2/"):
    categories = ['Bronchial', 'pneumonia', 'asthma', 'healthy', 'copd']

    # Load dataset and encode
    (X_train, y_train), (X_test, y_test), encoder = load_and_process_dataset(
        dataset_path=dataset_path,
        categories=categories
    )

    # Initialize model
    model = ConvLSTMNet(input_dim=13, num_classes=len(categories))

    if mode == "train":
        # Wrap and train
        train_dataset = LungSoundDataset(X_train, y_train)
        train_model(model, train_dataset, save_path=BEST_MODEL_FILE)

    elif mode == "evaluate":
        # Wrap and evaluate
        test_dataset = LungSoundDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=32)
        model.load_state_dict(torch.load(BEST_MODEL_FILE))
        evaluate_model(model, test_loader, encoder)

    else:
        raise ValueError("Mode must be either 'train' or 'evaluate'")


if __name__ == "__main__":
    args = get_args()

    main(mode=args.mode, dataset_path=args.data)
