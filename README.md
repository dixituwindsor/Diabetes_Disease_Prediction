# 🏥 Diabetes Disease Prediction using Machine Learning on Big Data

This repository contains a comprehensive project aimed at predicting diabetes disease using various **Machine Learning (ML)** models on large healthcare datasets. The project explores multiple algorithms to improve the accuracy and reliability of diabetes prediction, offering potential solutions for early diagnosis and treatment recommendations.

## 📁 Project Overview

The **Diabetes Disease Prediction System** uses machine learning algorithms to analyze healthcare data and predict whether a patient is likely to have diabetes. This study evaluates the effectiveness of multiple machine learning techniques and suggests improvements to the models for better performance.

Key objectives:
- Predict the presence of diabetes using patient health data.
- Compare and improve existing machine learning models.
- Address challenges like data security, privacy, and model accuracy.

## 🛠️ Features

- Implementation of several **machine learning models** to predict diabetes.
- Evaluation of models based on metrics like **Accuracy**, **Precision**, **Recall**, **F1-Score**, and **False Positive Rate**.
- Comprehensive comparison of models, including both traditional algorithms and advanced techniques like **XGBoost** and **CatBoost**.

## ⚙️ Machine Learning Models Used

The following machine learning models have been implemented and evaluated in this project:

- **Fully Connected Neural Network (FCNN)**
- **Long Short-Term Memory (LSTM)**
- **K-Nearest Neighbours (KNN)**
- **Logistic Regression**
- **Support Vector Machine (SVM)**
- **Random Forest Classification**
- **Naive Bayes Classification**
- **Simple CART (Decision Tree)**
- **XGBoost**
- **CatBoost**

Each model has been evaluated based on its accuracy and other key performance metrics to determine the most effective algorithm for diabetes prediction.

### 🧠 Model Comparison

| **Model**            | **Accuracy** | **Recall** | **False Positive Rate** | **Precision** | **F-Measure** |
|----------------------|--------------|------------|-------------------------|---------------|---------------|
| CNN                  | 72.08%       | 58.18%     | 20.20%                  | 61.54%        | 59.81%        |
| LSTM                 | 74.03%       | 70.91%     | 24.24%                  | 61.90%        | 66.10%        |
| KNN                  | 70.78%       | 50.91%     | 18.18%                  | 60.87%        | 55.45%        |
| Logistic Regression   | 75.32%       | 67.27%     | 20.20%                  | 64.91%        | 66.07%        |
| SVM                  | 75.97%       | 65.45%     | 18.18%                  | 66.67%        | 66.06%        |
| Random Forest         | 72.08%       | 61.82%     | 22.22%                  | 60.71%        | 61.26%        |
| Naive Bayes           | 76.62%       | 70.91%     | 20.20%                  | 66.10%        | 68.42%        |
| Simple CART           | 74.68%       | 72.73%     | 24.24%                  | 62.50%        | 67.23%        |
| XGBoost               | 68.83%       | 65.45%     | 29.29%                  | 55.38%        | 60.00%        |
| CatBoost              | 75.32%       | 67.27%     | 20.20%                  | 64.91%        | 66.07%        |

## 🚀 Technologies Used

- **Python** for data processing and machine learning.
- **NumPy**, **Pandas** for data manipulation and preprocessing.
- **Scikit-learn**, **XGBoost**, **CatBoost** for implementing machine learning algorithms.
- **Keras** for building and training deep learning models.
- **Matplotlib**, **Seaborn** for data visualization.
- **Dataspell IDE** for development and testing.


## 📊 Data Preprocessing

The dataset is preprocessed to ensure that the machine learning models receive clean and normalized input. The steps include:
1. **Data Cleaning**: Handling missing values and incorrect data entries.
2. **Feature Scaling**: Standardizing the data to a uniform scale for better model performance.
3. **Train/Test Split**: Dividing the data into 80% training and 20% testing sets for evaluation.

## 🔍 Performance Evaluation

The models were evaluated using several key metrics:
- **Accuracy**: The overall correctness of the model.
- **Precision**: The proportion of true positives among all positive predictions.
- **Recall (True Positive Rate)**: The proportion of actual positives that are correctly identified.
- **F-Measure**: The harmonic mean of precision and recall.
- **False Positive Rate**: The proportion of false positives among all negative cases.
