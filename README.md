# 💳 Credit Card Fraud Detection 🔍

Detecting fraudulent credit card transactions using **Machine Learning** and deploying it via an interactive **Streamlit** web app.

---

## 📌 Project Overview

Credit card fraud is a major concern in the financial sector, with fraud cases often being **rare but costly**. This project leverages **XGBoost**, a powerful gradient boosting classifier, to **identify suspicious transactions** effectively, even in highly imbalanced datasets.

We provide a **real-time prediction interface** built using **Streamlit**, enabling end-users to test transactions for potential fraud in just a few clicks!

---

## 🚀 Features

✅ Preprocesses and balances imbalanced data (fraud < 0.2%)  
✅ Trains an **XGBoost** model on the refined dataset  
✅ Evaluates with multiple robust metrics  
✅ Interactive **Streamlit UI** for real-time fraud checks  
✅ Saves and loads the trained model using `joblib`

---

## 🔧 How It Works

### 🧹 1. Data Preprocessing
- 📉 **Class imbalance** addressed with **Random Under-Sampling**  
- 🔀 **Train/Test Split**: Ensures fair model evaluation  
- 🧮 Selected features normalized/scaled if necessary

### 🧠 2. Model Training
- ✅ Model: **XGBoostClassifier**  
- 🧪 Evaluated using:
  - 🔍 Accuracy  
  - 🎯 Precision  
  - 🔁 Recall  
  - 🧩 F1-score  
  - 📈 ROC-AUC  

### 🌐 3. Deployment
- 💾 Model serialized using `joblib`  
- 🖥️ **Streamlit app** takes custom user input and returns predictions live  
- 🔐 Optional: Add authentication or logging for production deployment

---

## 🖼️ Streamlit App Preview

> 💡 *Users input transaction details, click a button, and receive instant predictions.*

📷 *(Insert screenshot of the app here)*

---

## 📁 Project Structure

credit-card-fraud-detection/
├── data/
│ └── creditcard.csv
├── model/
│ └── xgboost_model.pkl
├── app/
│ └── fraud_app.py
├── utils/
│ └── preprocessing.py
├── requirements.txt
└── README.md



## 🛠️ Technologies Used

- 🐍 Python 3.x
- 📊 Pandas, NumPy
- 🎯 Scikit-learn, XGBoost
- 🌐 Streamlit
- 💾 Joblib

---

## 📈 Model Performance (Example)

| Metric     | Score    |
|------------|----------|
| Accuracy   | 99.4%    |
| Precision  | 92.1%    |
| Recall     | 88.7%    |
| F1-score   | 90.3%    |
| ROC-AUC    | 97.5%    |

> ⚠️ *Exact scores may vary depending on data splits and hyperparameter tuning.*

---

## 🧪 Getting Started

### 🔨 Install Dependencies

```bash
pip install -r requirements.txt
streamlit run app/fraud_app.py
