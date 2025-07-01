# 🧠 Autism Prediction using Machine Learning

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A machine learning-based system to predict the likelihood of Autism Spectrum Disorder (ASD) based on questionnaire responses and user demographic data. This project uses multiple models and compares their performance.

---

## 🧾 Project Overview

Autism Spectrum Disorder (ASD) is a developmental disorder that affects communication and behavior. Early detection can help initiate timely intervention. This project builds a predictive system using classification models trained on structured data from questionnaire responses.

---

## 📁 Dataset

- **File**: `train.csv`
- **Attributes**: Age, Gender, Ethnicity, Family History, Symptoms, and Test Scores
- **Target**: Binary indicator of ASD diagnosis (1 = ASD, 0 = No ASD)

---

## 💡 Key Features

- Data preprocessing: handling null values, encoding categorical variables
- Multiple classification models: Logistic Regression, Random Forest, KNN, etc.
- Best model saved as `best_model.pkl`
- Encoders stored as `encoder.pkl`
- Performance comparison with accuracy, precision, recall, and F1-score

---

## 🧠 Machine Learning Models Used

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- Naive Bayes
- Gradient Boosting

---

## 📈 Model Performance

- Best performing model: Random Forest Classifier
- Accuracy: ~94% *(based on your notebook evaluation cell)*
- Metrics evaluated: Accuracy, Confusion Matrix, Classification Report

---

## 🛠️ Technologies Used

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Matplotlib / Seaborn
- Jupyter Notebook

---

## 📂 Project Structure

```
Autism-Prediction/
├── autism_prediction T4.ipynb   # Final model notebook
├── autism_prediction.ipynb      # Initial notebook version (optional)
├── train.csv                    # Dataset
├── best_model.pkl               # Trained model
├── encoder.pkl                  # Encoder for categorical values
├── requirements.txt             # Dependencies file
├── LICENSE                      # license file
└── README.md                    # Project overview
```

---

## 🚀 Getting Started

### 1. Clone the repository

```
git clone https://github.com/yourusername/Autism-Prediction.git
cd Autism-Prediction
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run the notebook

```
jupyter notebook "autism_prediction T4.ipynb"
```

## 🤝 Contributing
Contributions, suggestions, and improvements are welcome!
Open an issue or submit a pull request.

## 📝 License
This project is licensed under the MIT License.

## 🙌 Acknowledgments
Autism screening data sources

Scikit-learn and open-source ML libraries
