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

import torch
from torch.utils.data import Dataset

class LungSoundDataset(Dataset):
    def __init__(self, X, y):
        """
        Args:
            X (np.ndarray): Input features, shape (num_samples, timesteps, 1)
            y (np.ndarray): Encoded labels
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]