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

# Base output folder
BASE_OUTPUT_PATH = "./output"

# Subfolders
MODEL_PATH = os.path.join(BASE_OUTPUT_PATH, "model")
PLOTS_PATH = os.path.join(BASE_OUTPUT_PATH, "plots")
METRICS_PATH = os.path.join(BASE_OUTPUT_PATH, "metrics")
RECORDS_PATH = os.path.join(BASE_OUTPUT_PATH, "records")

# Ensure directories exist
for path in [MODEL_PATH, PLOTS_PATH, METRICS_PATH]:
    os.makedirs(path, exist_ok=True)

# Default best model save path
BEST_MODEL_FILE = os.path.join(MODEL_PATH, "model_best.pth")