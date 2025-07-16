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
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize

from utils.config import PLOTS_PATH, METRICS_PATH

def evaluate_model(model, test_loader, encoder, device="cuda" if torch.cuda.is_available() else "cpu"):
    os.makedirs(PLOTS_PATH, exist_ok=True)
    os.makedirs(METRICS_PATH, exist_ok=True)

    model = model.to(device)
    model.eval()

    y_true = []
    y_pred = []
    y_proba = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probs = outputs.cpu().numpy()
            preds = np.argmax(probs, axis=1)

            y_proba.append(probs)
            y_pred.extend(preds)
            y_true.extend(y_batch.numpy())

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    y_proba = np.vstack(y_proba)

    # Accuracy
    acc = (y_pred == y_true).mean()
    print(f"\n✅ Test Accuracy: {acc * 100:.2f}%")

    # Save classification report
    report = classification_report(y_true, y_pred, target_names=encoder.classes_)
    report_file = os.path.join(METRICS_PATH, "classification_report.txt")
    with open(report_file, "w") as f:
        f.write(f"Test Accuracy: {acc * 100:.2f}%\n\n")
        f.write(report)

    # Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    plt.rcParams['figure.facecolor'] = 'w'
    plt.rcParams['axes.facecolor'] = 'w'
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title("Confusion Matrix (Test Data)", color='black')
    plt.xlabel("Predicted Label", color='black')
    plt.ylabel("True Label", color='black')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_PATH, "confusion_matrix.png"))
    plt.close()

    # AUC-ROC
    n_classes = len(encoder.classes_)
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    fpr, tpr, roc_auc = {}, {}, {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(6, 5))
    colors = ['red', 'green', 'blue', 'orange', 'purple']
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'{encoder.classes_[i]} (AUC={roc_auc[i]:.2f})', color=colors[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate', color='black')
    plt.ylabel('True Positive Rate', color='black')
    plt.title('Multi-Class AUC-ROC', color='black')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_PATH, "auc_roc_curve.png"))
    plt.close()