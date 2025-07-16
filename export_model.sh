#!/bin/bash

echo "📦 Exporting model from PyTorch to TorchScript..."

python - <<EOF
import torch
from model.network import ConvLSTMNet
from utils.config import BEST_MODEL_FILE, MODEL_PATH
import os

# Initialize model
model = ConvLSTMNet(input_dim=13, num_classes=5)
model.load_state_dict(torch.load(BEST_MODEL_FILE))
model.eval()

# Example dummy input (batch_size=1, timesteps=13, channels=1)
dummy_input = torch.randn(1, 13, 1)

# Export to TorchScript
scripted_model = torch.jit.trace(model, dummy_input)
export_path = os.path.join(MODEL_PATH, "model_scripted.pt")
scripted_model.save(export_path)

print(f"✅ TorchScript model saved at: {export_path}")
EOF
