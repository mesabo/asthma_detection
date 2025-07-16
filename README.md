# Asthma Detection from Lung Audio

This PyTorch-based project detects asthma from lung sound recordings with deep learning.
It supports real-time input, feature extraction, and deployment capabilities.

Dataset available here
[kaggle: Asthma Detection from Lung Audio Dataset](https://www.kaggle.com/code/rashidul0/asthma-detection-from-lung-audio-dataset)
## 🛠 Setup (Anaconda)
```bash
conda env create -f audio_env.yaml
conda activate audio_env
```

```bash
pip3 install torch torchvision torchaudio
```

## 📁 Structure
- `data/`: Raw and processed audio
- `utils/`: Augmentation, feature extraction, config, dataset
- `model/`: PyTorch model, training, and evaluation
- `real_time/`: Live audio input and prediction
- `deployment/`: Streamlit or FastAPI app
- `notebooks/`: EDA and analysis
- `main.py`: Entry point

## 🚀 Run
```bash
python main.py
```
