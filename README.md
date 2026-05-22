# 🧠 Solving the Vanishing Gradient Problem with LSTM

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.16-orange)
![Streamlit](https://img.shields.io/badge/streamlit-1.32-red)

An end-to-end, production-ready Deep Learning project demonstrating the theoretical and practical differences between Simple Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks in solving the Vanishing Gradient Problem.

---

## 🎯 Project Objectives
1. **Understand Theoretical Flaws:** Demonstrate mathematically and practically why Simple RNNs fail at long-term dependencies (Vanishing Gradient).
2. **Implement the Solution:** Show how LSTM's cell state and gate mechanisms solve this problem.
3. **Build an Application:** Create a Context-Aware Next Word Prediction System comparing both architectures in real-time.

## 📁 Folder Structure
```
VanishX/
│
├── data/                   # Downloaded dataset (Shakespeare text)
├── docs/                   # Complete documentation (Report, PPT, Resume points)
├── models/                 # Saved .h5 models, tokenizer, and metadata
├── visualizations/         # Accuracy and Loss comparison graphs
│
├── train.py                # Data prep, Model training, and evaluation script
├── app.py                  # Streamlit Frontend application
├── requirements.txt        # Project dependencies
└── README.md               # You are here
```

## 🚀 Quick Start (Local Setup)

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/VanishX.git
cd VanishX
```

### 2. Create Virtual Environment
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Models
*(This will download the data, preprocess it, train the RNN and LSTM, and generate graphs)*
```bash
python train.py
```

### 5. Run the Streamlit Web App
```bash
streamlit run app.py
```

## 📊 Results & Comparative Analysis
The project trains on Shakespearean text. As shown in the generated `visualizations/` folder:
- **RNN** struggles to maintain context beyond a few words, resulting in higher validation loss and nonsensical predictions for longer phrases.
- **LSTM** effectively retains memory using its Forget, Input, and Output gates, yielding higher accuracy and grammatically/contextually superior text generation.

## 🌐 Deployment
This project is fully compatible with Streamlit Cloud, Render, and Hugging Face Spaces. See `docs/Deployment_Guide.md` for step-by-step instructions.

## 👨‍💻 Author
**Your Name**
- LinkedIn: [Your Profile]
- GitHub: [Your Profile]
