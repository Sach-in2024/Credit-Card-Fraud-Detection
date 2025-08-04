# 💳 Credit Card Fraud Detection

A machine learning project that identifies fraudulent credit card transactions using real-world imbalanced data. The goal is to help financial institutions detect suspicious behavior with high accuracy and minimal false positives.

## 📌 Problem Statement

Credit card fraud is a significant financial problem. Due to the class imbalance (fraudulent vs. non-fraudulent), it's a challenging task for traditional machine learning models. This project applies data preprocessing, resampling techniques, and several ML models to accurately detect fraud.

## 🚀 Technologies Used

- **Python 3.8+**
- **Pandas** – data analysis
- **NumPy** – numerical computations
- **Matplotlib & Seaborn** – data visualization
- **Scikit-learn** – ML models
- **Imbalanced-learn** – handling imbalanced datasets (SMOTE, RandomUnderSampler)

## 🧠 ML Models Applied

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost (Optional for performance)
- Model Evaluation: Confusion Matrix, Precision, Recall, F1-score, ROC-AUC

## 📊 Dataset

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Size**: 284,807 transactions
- **Features**: 30 (anonymized using PCA)
- **Class Imbalance**: Only ~0.17% fraud cases

## 🛠️ How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
