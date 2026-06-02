<div align="center">

# 🌙 SleepSense AI

### Deep Learning · Sleep Disorder Prediction · 49-Feature Analysis

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Keras](https://img.shields.io/badge/Keras-Sequential-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

A beautiful, production-grade **Streamlit web application** that uses a pre-trained deep learning model to predict sleep disorders — **No Disorder**, **Sleep Apnea**, or **Insomnia** — from 49 physiological and lifestyle features.

![SleepSense AI Banner](https://img.shields.io/badge/UI-Dark%20Cosmic%20Theme-6366f1?style=flat-square) ![Model](https://img.shields.io/badge/Model-Sequential%20DNN-0ea5e9?style=flat-square) ![Classes](https://img.shields.io/badge/Classes-3%20Disorders-f59e0b?style=flat-square)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Model Architecture](#-model-architecture)
- [Input Features](#-input-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Disclaimer](#️-disclaimer)

---

## 🔭 Overview

SleepSense AI is an end-to-end machine learning web application for clinical sleep disorder screening. It wraps a trained Keras Sequential neural network inside a polished, dark-themed Streamlit UI that allows users to enter a full patient profile across 8 clinical categories and instantly receive a probability-scored prediction with a clinical interpretation.

**Predicted Classes:**

| Class | Icon | Description |
|---|---|---|
| No Disorder | ✅ | Healthy sleep pattern detected |
| Sleep Apnea | 😮‍💨 | Indicators of obstructive or central sleep apnea |
| Insomnia | 😴 | Features consistent with chronic insomnia |

---

## ✨ Features

- 🎨 **Dark cosmic UI** — gradient mesh background, glassmorphism cards, animated hover states
- 🧠 **Deep learning inference** — real-time predictions using a 4-layer dense neural network
- 📊 **Probability distribution** — gradient progress bars showing confidence per class
- 🩺 **Clinical interpretation** — urgency badge + tailored recommendation per prediction
- 📋 **49-feature form** — logically grouped into 8 clinical input sections
- 📌 **Key metrics summary** — at-a-glance tiles for the most important patient vitals
- ⚡ **Instant results** — sub-second inference with smooth loading animation
- 📱 **Wide layout** — optimised for desktop clinical workstations

---

## 🧠 Model Architecture

```
Input Layer  →  49 features (float32)
Dense        →  50 units  | activation: linear
Dense        →  128 units | activation: ReLU
Dense        →  64 units  | activation: ReLU
Dense        →  32 units  | activation: ReLU
Output       →  3 units   | activation: Sigmoid
```

| Hyperparameter | Value |
|---|---|
| Loss Function | Sparse Categorical Crossentropy |
| Optimizer | Adam (lr = 0.001) |
| Metrics | Accuracy |
| Input Shape | (None, 49) |
| Output Classes | 3 |
| File Format | HDF5 (`.h5`) |

---

## 🗂️ Input Features

The model accepts **49 numerical/encoded features** grouped into 8 clinical categories:

<details>
<summary><b>👤 Demographics & BMI</b> (3 features)</summary>

| Feature | Type | Range |
|---|---|---|
| Age | Integer | 18 – 90 |
| Gender | Binary | Male / Female |
| BMI | Float | 15.0 – 50.0 |

</details>

<details>
<summary><b>🛏️ Sleep Metrics</b> (4 features)</summary>

| Feature | Type | Range |
|---|---|---|
| Sleep Duration (hrs) | Float | 2 – 12 |
| Sleep Quality (1–10) | Integer | 1 – 10 |
| Sleep Latency (min) | Integer | 0 – 120 |
| Night Wake-ups | Integer | 0 – 10 |

</details>

<details>
<summary><b>💓 Vitals & Activity</b> (4 features)</summary>

| Feature | Type | Range |
|---|---|---|
| Heart Rate (bpm) | Integer | 40 – 120 |
| SpO₂ (%) | Integer | 85 – 100 |
| Blood Pressure (systolic) | Integer | 80 – 200 |
| Daily Steps | Integer | 0 – 20,000 |

</details>

<details>
<summary><b>🌿 Lifestyle & Stress</b> (6 features)</summary>

| Feature | Type | Range |
|---|---|---|
| Stress Level (1–10) | Integer | 1 – 10 |
| Caffeine Intake (mg/day) | Integer | 0 – 600 |
| Alcohol Units/Week | Integer | 0 – 21 |
| Smoking Status | Ordinal | Non-smoker / Former / Current |
| Exercise (hrs/week) | Float | 0 – 14 |
| Screen Time (hrs/day) | Float | 0 – 16 |

</details>

<details>
<summary><b>🏥 Medical Conditions</b> (8 features)</summary>

Anxiety Disorder, Depression, Hypertension, Diabetes, Asthma, Heart Disease, Snoring, Restless Legs — each encoded as binary (0 = No, 1 = Yes).

</details>

<details>
<summary><b>🌡️ Environment & Medications</b> (6 features)</summary>

Room Temperature, Noise Level, Light Level, Sleep Medications, Shift Work, Daytime Naps.

</details>

<details>
<summary><b>📊 Clinical Measures</b> (9 features)</summary>

AHI Score, Epworth Sleepiness Score (ESS), PSQI Score, REM Sleep %, Deep Sleep %, Sleep Efficiency %, Chronotype, Work Hours/Day, Social Jetlag.

</details>

<details>
<summary><b>💼 Occupation</b> (1 feature)</summary>

Encoded as ordinal: Office Worker, Healthcare, Student, Manual Labour, Night Shift Worker, Retired, Other.

</details>

---

## ⚙️ Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-username/sleepsense-ai.git
cd sleepsense-ai

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### requirements.txt

```text
streamlit>=1.28.0
tensorflow>=2.12.0
numpy>=1.23.0
```

---

## 🚀 Usage

```bash
streamlit run app.py
```

Then open your browser at **http://localhost:8501**

1. Fill in the patient profile across all 8 input sections
2. Click **🔮 Analyse Sleep Pattern**
3. View the prediction, probability bars, and clinical interpretation

---

## 📁 Project Structure

```
sleepsense-ai/
│
├── app.py                      # Main Streamlit application
├── sleep_disorder_model.h5     # Pre-trained Keras model (HDF5)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend UI | Streamlit + custom CSS |
| Deep Learning | TensorFlow / Keras |
| Numerical Computing | NumPy |
| Fonts | Sora, JetBrains Mono (Google Fonts) |
| Model Format | Keras HDF5 (`.h5`) |

---

## ⚕️ Disclaimer

> **This application is intended for educational and research purposes only.**
> Predictions generated by this model should **not** be used as a substitute for professional medical diagnosis, advice, or treatment. Always consult a qualified and licensed healthcare professional for any sleep-related health concerns.

---

<div align="center">

Made with 🌙 and deep learning

⭐ Star this repo if you found it useful!

</div>
