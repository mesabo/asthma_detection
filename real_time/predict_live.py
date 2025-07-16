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

# Real-time prediction using trained model

import torch
import librosa
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.config import BEST_MODEL_FILE, RECORDS_PATH
from model.network import ConvLSTMNet
from real_time.record_audio import record_audio

# Categories used during training
categories = ['Bronchial', 'pneumonia', 'asthma', 'healthy', 'copd']
encoder = LabelEncoder()
encoder.fit(categories)

def extract_mfcc_from_file(filepath, n_mfcc=13):
    y, sr = librosa.load(filepath, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    features = np.mean(mfcc, axis=1).reshape(1, n_mfcc, 1)
    return torch.tensor(features, dtype=torch.float32)

def predict_live():
    # Step 1: Record and store under ../output/records/
    filename = os.path.join("../", RECORDS_PATH, "recorded.wav")
    record_audio(filename=filename, duration=5)

    # Step 2: Load model from ../output/model/
    model_path = os.path.join("../", BEST_MODEL_FILE)
    model = ConvLSTMNet(input_dim=13, num_classes=len(categories))
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    # Step 3: Feature extraction and prediction
    x = extract_mfcc_from_file(filename)
    with torch.no_grad():
        out = model(x)
        pred = torch.argmax(out, dim=1).item()

    label = encoder.inverse_transform([pred])[0]
    print(f"🩺 Predicted Condition: **{label}**")

if __name__ == "__main__":
    predict_live()