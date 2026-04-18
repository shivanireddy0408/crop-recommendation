# 🌾 Smart Crop Recommendation System

> *AI-powered precision agriculture — recommending the right crop for your soil and climate.*

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red?logo=streamlit)
![Accuracy](https://img.shields.io/badge/Accuracy-98.6%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Project Overview

The **Smart Crop Recommendation System** is an end-to-end Machine Learning application that assists
farmers and agronomists in selecting the most suitable crop for a given field. It takes soil nutrient
levels and local climate conditions as input and outputs the best-matched crop — along with the model's
confidence score and top-5 alternatives.

The system is backed by a **Random Forest classifier** achieving **98.6% accuracy** across 22 crop
classes, and is served through a polished **Streamlit web application**.

---

## ❓ Problem Statement

Agriculture is the backbone of many economies, yet crop selection is still largely driven by tradition
or guesswork. Choosing the wrong crop for a given soil type and climate can lead to:

- Poor yields and financial loss for farmers
- Excessive use of fertilisers and water
- Environmental degradation

**Machine Learning offers a data-driven solution**: by learning from historical agronomic data, a model
can recommend the optimal crop with high accuracy — personalised to each field's unique conditions.

---

## 📊 Dataset

| Property | Value |
|---|---|
| Rows | 2,200 |
| Crops (classes) | 22 |
| Samples per crop | 100 |
| Missing values | None |
| Source | Agronomic profiles (synthetic, based on real ranges) |

### Features

| Feature | Description | Unit |
|---|---|---|
| `N` | Nitrogen content in soil | kg/ha |
| `P` | Phosphorus content in soil | kg/ha |
| `K` | Potassium content in soil | kg/ha |
| `temperature` | Average ambient temperature | °C |
| `humidity` | Relative humidity | % |
| `ph` | Soil pH value | — |
| `rainfall` | Average annual rainfall | mm |

### Target

| Column | Description |
|---|---|
| `label` | Recommended crop name (22 classes) |

---

## 🤖 ML Approach

### Pipeline

```
Raw CSV
  │
  ├─ Null check & descriptive statistics
  ├─ LabelEncoder  → encode crop names
  ├─ StandardScaler → feature normalisation (Logistic Regression only)
  ├─ Stratified 80/20 train–test split
  │
  ├─ Train  Logistic Regression
  ├─ Train  Decision Tree
  └─ Train  Random Forest   ← best model
             │
             ├─ Classification report
             ├─ Confusion matrix
             ├─ Feature importance
             └─ Save best_model.pkl
```

### Preprocessing

- **No missing values** — data is clean.
- **LabelEncoder** for the target (`label` → integer class).
- **StandardScaler** fitted on training data and applied to test data (used for Logistic Regression;
  tree-based models receive unscaled features).
- **Stratified split** ensures equal class representation in train and test sets.

---

## 📈 Model Comparison

| Model | Test Accuracy | CV Mean (5-fold) | CV Std |
|---|---|---|---|
| **Random Forest ✅** | **98.64%** | 96.82% | ±0.99% |
| Logistic Regression | 97.73% | 96.53% | ±0.97% |
| Decision Tree | 94.32% | 92.78% | ±0.93% |

**Best model**: Random Forest (200 estimators, max_depth=15)

### Key Findings

- Random Forest outperforms all other models on both test accuracy and cross-validation.
- `K` (Potassium), `rainfall`, and `humidity` are the three most important features.
- Crops with overlapping climate requirements (e.g. jute/rice) are the most challenging to separate.

---

## 🎯 Results

### Classification Report (Best Model — Random Forest)

```
              precision    recall  f1-score
apple             1.00      1.00      1.00
banana            1.00      0.95      0.97
blackgram         1.00      1.00      1.00
chickpea          1.00      1.00      1.00
coconut           1.00      1.00      1.00
coffee            1.00      1.00      1.00
cotton            1.00      1.00      1.00
grapes            1.00      1.00      1.00
jute              0.83      0.95      0.88
kidneybeans       1.00      1.00      1.00
lentil            1.00      0.95      0.97
maize             1.00      1.00      1.00
mango             1.00      1.00      1.00
mothbeans         0.95      1.00      0.98
mungbean          1.00      1.00      1.00
muskmelon         1.00      1.00      1.00
orange            1.00      1.00      1.00
papaya            1.00      1.00      1.00
pigeonpeas        1.00      1.00      1.00
pomegranate       1.00      1.00      1.00
rice              0.94      0.85      0.89
watermelon        1.00      1.00      1.00

accuracy                           0.99 (440 samples)
```

---

## 🌐 Web Application

The Streamlit app provides:

- **7 interactive sliders** for soil nutrients (N, P, K), climate (temperature, humidity, rainfall)
  and soil chemistry (pH).
- **Recommended crop** with emoji and confidence score.
- **Visual confidence bar** (green ≥ 70 %, amber < 70 %).
- **Top-5 alternatives** with probabilities.
- **Input summary table**.
- **EDA tab** showing all training charts.
- **About tab** with model details and run instructions.

---

## 📁 Project Structure

```
crop-recommendation/
│
├── data/
│   ├── generate_dataset.py     ← generates synthetic dataset
│   └── Crop_recommendation.csv ← generated dataset (2 200 rows)
│
├── notebook/                   ← EDA & model charts (auto-generated)
│   ├── correlation_heatmap.png
│   ├── feature_distributions.png
│   ├── boxplots_by_crop.png
│   ├── model_comparison.png
│   ├── feature_importance.png
│   └── confusion_matrix.png
│
├── src/
│   └── train.py                ← full ML pipeline
│
├── model/                      ← saved artefacts (auto-generated)
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   └── model_summary.txt
│
├── app.py                      ← Streamlit web application
├── requirements.txt
└── README.md
```

---

## ⚙️ How to Run Locally

### Prerequisites

- Python 3.9+
- pip

### Step-by-step

```bash
# 1. Clone the repository
git clone https://github.com/shivanireddy0408/crop-recommendation.git
cd crop-recommendation

# 2. (Recommended) Create a virtual environment
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Generate the dataset
python data/generate_dataset.py

# 5. Train all models, produce EDA charts, and save the best model
python src/train.py

# 6. Launch the Streamlit app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 🖼️ Screenshots

> _Add screenshots here after running the app._

| Prediction Tab | EDA Charts |

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data loading & manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | ML models, preprocessing, metrics |
| `matplotlib` | Static charts |
| `seaborn` | Statistical visualisations |
| `streamlit` | Interactive web application |
| `joblib` | Model serialisation |

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute.

---

## 🙏 Acknowledgements

- Agronomic data ranges sourced from FAO crop production guidelines.
- Inspired by the Kaggle *Crop Recommendation Dataset* community.

---

*Built with ❤️ using Python · scikit-learn · Streamlit*
