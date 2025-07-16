#!/bin/bash
#SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH --nodelist=ai-gpgpu14      # Node to run the job

# Load user environment
source ~/.bashrc
hostname

# Detect computation device
DEVICE="cpu"
if [[ -n "$CUDA_VISIBLE_DEVICES" && $(nvidia-smi | grep -c "GPU") -gt 0 ]]; then
    DEVICE="cuda"
elif [[ "$(uname -s)" == "Darwin" && $(sysctl -n machdep.cpu.brand_string) == *"Apple"* ]]; then
    DEVICE="mps"
fi

# Activate Conda environment
ENV_NAME="audio_env"
if [[ -z "$CONDA_DEFAULT_ENV" || "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]]; then
    if ! command -v conda &>/dev/null; then
        echo "Error: Conda not found. Please install Conda and ensure it's in your PATH."
        exit 1
    fi
    source activate "$ENV_NAME" || { echo "Error: Could not activate Conda environment '$ENV_NAME'."; exit 1; }
fi

# Configurations
MODES=("train" "evaluate")
for MODE in "${MODES[@]}"; do
  python -u main.py \
  --mode "$MODE"
done

echo "🌟 Execution Complete 🌟"
