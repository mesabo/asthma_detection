# 🩺 Asthma Detection from Lung Audio (PyTorch)

This project leverages **deep learning with PyTorch** to detect asthma and other respiratory conditions from lung sound recordings. It supports:

- 🔊 Audio preprocessing & augmentation  
- 🧠 Deep neural network (Conv1D + LSTM) training  
- 📊 Model evaluation (Accuracy, Confusion Matrix, AUC-ROC)  
- 🎤 Real-time inference  
- 📦 Deployment-ready code (Streamlit / FastAPI)

---

## 📦 Dataset

Download the dataset from Kaggle:
👉 [Asthma Detection from Lung Audio Dataset](https://www.kaggle.com/code/rashidul0/asthma-detection-from-lung-audio-dataset)

Place it in:  
```bash
./Dataset/Asthma Detection Dataset Version 2/
```

---

## 🛠 Setup

### ✅ Option 1: Using Anaconda (Recommended)
```bash
conda env create -f audio_env.yaml
conda activate audio_env
```

> 🧠 Then install PyTorch:
```bash
pip install torch torchvision torchaudio
```

---

### 🐍 Option 2: Using virtualenv + `requirements.txt`
```bash
python3 -m venv audio_env
source audio_env/bin/activate  # On Windows: audio_env\Scripts\activate
pip install -r requirements.txt
```

---

## 📁 Project Structure

```
asthma_detector/
├── Dataset/                      # Raw input data
├── audio_env.yaml                # Conda environment
├── requirements.txt              # venv dependency list
├── main.py                       # Entry point (train/evaluate)
├── run.sh                        # Train & evaluate automation
├── build_and_export.sh           # Export the trained model
├── model/
│   ├── network.py                # ConvLSTM network
│   ├── train.py                  # Training logic
│   └── evaluate.py               # Evaluation logic
├── real_time/                    # Real-time inference scripts
├── utils/
│   ├── audio_processing.py       # Augmentation & preprocessing
│   ├── dataset.py                # PyTorch Dataset class
│   ├── config.py                 # Paths and constants
│   └── argument_parser.py        # Argument parsing
└── output/
    ├── model/                    # Trained model
    ├── logs/                     # Logs and metrics
    ├── plots/                    # Confusion matrix, ROC, etc.
```

---

## 🚀 How to Run

### 1. **Train the model**
```bash
python main.py --mode train --data "./Dataset/Asthma Detection Dataset Version 2/"
```

### 2. **Evaluate the model**
```bash
python main.py --mode evaluate --data "./Dataset/Asthma Detection Dataset Version 2/"
```

### 3. **Run both (auto)**
```bash
bash run.sh
```

---

## 🎤 Real-Time Inference
Capture live audio and classify:
```bash
python real_time/predict_live.py
```

---

## 🧪 Export Trained Model
```bash
bash build_and_export.sh
```
Trained model will be saved to:
```
output/model/best_model.pth
```

---

## 👤 Author

- **Name:** Franck Junior Aboya Messou  
- **Email:** [messouaboya17@gmail.com](mailto:messouaboya17@gmail.com)  
- **GitHub:** [mesabo](https://github.com/mesabo)  
- **University:** HOSEI University, Japan  
- **Lab:** Prof. YU Keping’s Lab, IIST  

---

## 📜 License
This project is for educational and research purposes only.
