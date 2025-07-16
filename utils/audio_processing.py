#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025/07/17

🚀 Welcome to the Awesome Python Script 🚀

User: Messou Franck Junior Aboya
Email: messouaboya17@gmail.com
Github: https://github.com/mesabo
Univ: Hosei University - IIST - (Tokyo, Japan)
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

import os
import numpy as np
import librosa
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# Audio Augmentation Functions
# -------------------------------
def my_add_noise(audio, noise_factor=0.005):
    noise = np.random.randn(len(audio))
    return audio + noise_factor * noise

def my_time_stretch(audio, rate=1.0):
    return librosa.effects.time_stretch(y=audio, rate=rate)

def my_pitch_shift(audio, sr, n_steps=0):
    return librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=n_steps)

def augment_audio(y, sr):
    choice = random.choice(['noise', 'stretch', 'pitch', 'none'])
    if choice == 'noise':
        return my_add_noise(y)
    elif choice == 'stretch':
        rate = random.uniform(0.8, 1.2)
        return my_time_stretch(y, rate=rate)
    elif choice == 'pitch':
        steps = random.uniform(-2, 2)
        return my_pitch_shift(y, sr, n_steps=steps)
    return y

# -------------------------------
# Feature Extraction
# -------------------------------
def extract_mfcc(y, sr, n_mfcc=13):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)  # shape: (n_mfcc,)

# -------------------------------
# Main Data Processing Function
# -------------------------------
def load_and_process_dataset(dataset_path, categories, test_size=0.2, n_mfcc=13):
    print("📁 Loading and processing dataset...")
    all_audio_data = []
    category_counts = {}

    # Count original samples per category
    for category in categories:
        cat_path = os.path.join(dataset_path, category)
        count = len([f for f in os.listdir(cat_path) if f.endswith('.wav')])
        category_counts[category] = count

    max_count = max(category_counts.values())

    for category in categories:
        cat_path = os.path.join(dataset_path, category)
        audio_files = [f for f in os.listdir(cat_path) if f.endswith('.wav')]
        current_count = len(audio_files)

        loaded_signals = []
        for audio_file in audio_files:
            path = os.path.join(cat_path, audio_file)
            y, sr = librosa.load(path, sr=None)
            loaded_signals.append((y, sr))
            all_audio_data.append((y, sr, category))

        if current_count < max_count:
            needed = max_count - current_count
            for _ in range(needed):
                y_base, sr_base = random.choice(loaded_signals)
                y_aug = augment_audio(y_base, sr_base)
                all_audio_data.append((y_aug, sr_base, category))

    print("🔄 Extracting features...")
    features = []
    labels = []

    for y, sr, label in all_audio_data:
        feat = extract_mfcc(y, sr, n_mfcc=n_mfcc)
        features.append(feat)
        labels.append(label)

    X = np.array(features)
    y = np.array(labels)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Reshape for LSTM: (samples, timesteps, 1)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=42
    )

    print(f"✅ Done. Train: {len(X_train)}, Test: {len(X_test)}")
    return (X_train, y_train), (X_test, y_test), encoder