# Credit Card Defaulter Prediction Project Report

## 1. Introduction
The objective of this project is to build a predictive model to determine the likelihood of a credit card client defaulting on their payment. By analyzing demographic data and payment history, we aim to identify high-risk clients, which can help financial institutions in risk management and decision-making.

## 2. Dataset Overview
The dataset used for this analysis is Credit Card Defaulter Prediction.csv. It contains information on default payments, demographic factors, credit data, history of payment, and bill statements of credit card clients.

**Key Features:**
*   **Demographics:** SEX, EDUCATION, MARRIAGE, AGE
*   **Financial History:** PAY_0 to PAY_6 (Repayment status), BILL_AMT1 to BILL_AMT6 (Bill statement amount), PAY_AMT1 to PAY_AMT6 (Previous payment amount)
*   **Target Variable:** default (Yes/No) indicating whether the client defaulted.

## 3. Methodology

### 3.1 Data Preprocessing
To prepare the data for modeling, several preprocessing steps were undertaken:
*   **Data Cleaning:** The ID column was dropped as it provides no predictive value. Duplicate records were identified and removed to ensure data quality.
*   **Encoding:** Categorical variables such as SEX,  EDUCATION, MARRIAGE and the target default were encoded into numerical format to be compatible with machine learning algorithms.
*   **Train-Test Split:** The dataset was split into training (80%) and testing (20%) sets to evaluate model performance on unseen data.
*   **Handling Imbalance:** The dataset was found to be imbalanced (fewer defaulters than non-defaulters). **SMOTE (Synthetic Minority Over-sampling Technique)** was applied to the training data to balance the class distribution, ensuring the models don't become biased towards the majority class.
*   **Feature Scaling:** StandardScaler was used to scale the features, ensuring that all variables contribute equally to the model training process.

### 3.2 Exploratory Data Analysis (EDA)
Visualizations were created to understand the distribution of data and relationships between variables. This included analyzing the distribution of the target variable and demographic features.

### 3.3 Model Selection
Three different machine learning algorithms were implemented and evaluated:
1.  **Logistic Regression:** A linear model used as a baseline.
2.  **Random Forest Classifier:** An ensemble learning method that operates by constructing a multitude of decision trees.
3.  **XGBoost Classifier:** An optimized distributed gradient boosting library designed to be highly efficient, flexible, and portable.

## 4. Model Evaluation and Results

The models were evaluated based on **Accuracy**, **ROC-AUC Score**, **Precision**, **Recall**, and **F1-Score**.

### 4.1 Logistic Regression
*   **Accuracy:** 81.89%
*   **ROC-AUC:** 0.7012
*   **Performance:** Achieved the highest accuracy but the lowest ROC-AUC score. It is good for interpretability but struggles with complex non-linear patterns.

### 4.2 Random Forest Classifier
*   **Accuracy:** 80.66%
*   **ROC-AUC:** 0.7454
*   **Performance:** Showed a good balance, improving the ROC-AUC score significantly over Logistic Regression, indicating better capability in distinguishing between classes.

### 4.3 XGBoost Classifier
*   **Accuracy:** 81.65%
*   **ROC-AUC:** 0.7569
*   **Performance:** Achieved the best overall performance with a high accuracy comparable to Logistic Regression and the highest ROC-AUC score among all models.

### 4.4 Confusion Matrix Analysis
Confusion matrices were generated for each model to visualize the performance in terms of True Positives, True Negatives, False Positives, and False Negatives.
*   **XGBoost** demonstrated a robust ability to correctly identify defaulters (True Positives) while maintaining a high number of correct non-default predictions (True Negatives).

## 5. Conclusion
Based on the comprehensive evaluation:

*   **Logistic Regression** is suitable if model interpretability is the primary concern.
*   **Random Forest** captures non-linear relationships better than the linear model.
*   **XGBoost is the recommended model** for this project. It offers the best balance between accuracy and the ability to discriminate between defaulters and non-defaulters (highest ROC-AUC). It is robust and provides the most reliable predictions for identifying high-risk clients.
