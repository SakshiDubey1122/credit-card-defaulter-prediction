Credit Card Defaulter Prediction
Overview

This project focuses on predicting whether a credit card customer is likely to default on their payment. Using demographic information and historical payment behavior, multiple machine learning models are trained and compared to identify high-risk customers. The goal is to support better credit risk assessment and decision-making for financial institutions.

Dataset

The analysis uses Credit Card Defaulter Prediction.csv, which contains customer-level information related to credit usage and repayment behavior.

Key Feature Groups

Demographics

SEX

EDUCATION

MARRIAGE

AGE

Payment History

PAY_0 to PAY_6: Repayment status for previous months

BILL_AMT1 to BILL_AMT6: Monthly bill amounts

PAY_AMT1 to PAY_AMT6: Amounts paid in previous months

Target Variable

default (Yes / No), indicating whether the customer defaulted on their credit card payment.

Methodology
Data Preprocessing

Several preprocessing steps were applied before modeling:

Data Cleaning

Dropped the ID column as it does not contribute to prediction.

Checked for and removed duplicate records.

Encoding

Categorical variables such as SEX, EDUCATION, and MARRIAGE were encoded into numerical form.

The target variable (default) was converted into a binary format.

Train–Test Split

Data was split into 80% training and 20% testing sets using stratification to preserve class proportions.

Handling Class Imbalance

The dataset was imbalanced, with fewer defaulters than non-defaulters.

SMOTE (Synthetic Minority Over-sampling Technique) was applied only to the training data to balance the classes and prevent model bias.

Feature Scaling

StandardScaler was used to normalize feature values so that no variable dominated model training.

Exploratory Data Analysis (EDA)

EDA was performed to understand:

The distribution of the target variable

Demographic patterns

Relationships between financial behavior and default risk

Visualizations were generated to support these insights.

Models Implemented

Three supervised learning models were trained and evaluated:

Logistic Regression

Used as a baseline model for interpretability.

Random Forest Classifier

An ensemble method capable of capturing non-linear relationships.

XGBoost Classifier

A gradient boosting model optimized for performance and accuracy.

Model Evaluation and Results

Models were evaluated using:

Accuracy

ROC-AUC score

Precision, Recall, and F1-score

Confusion matrices

Logistic Regression

Accuracy: 81.89%

ROC-AUC: 0.7012

Performs well in terms of accuracy but has limited ability to capture complex patterns.

Random Forest

Accuracy: 80.66%

ROC-AUC: 0.7454

Improves discrimination between defaulters and non-defaulters compared to Logistic Regression.

XGBoost

Accuracy: 81.65%

ROC-AUC: 0.7569

Achieves the best overall performance with strong predictive power and balanced classification.

Confusion Matrix Analysis

Confusion matrices were generated for all models.
XGBoost showed the strongest ability to correctly identify defaulters while maintaining a high rate of correct non-default predictions.

Conclusion

Logistic Regression is suitable when interpretability is the main priority.

Random Forest handles non-linear relationships better and improves class separation.

XGBoost is the recommended model for this project, offering the best balance of accuracy and discrimination capability based on ROC-AUC.

Overall, the results demonstrate that advanced ensemble models provide more reliable predictions for identifying high-risk credit card customers.

Files in This Repository

Code.ipynb – complete data preprocessing, modeling, and evaluation pipeline

Credit Card Defaulter Prediction.csv – dataset

*.png – visualizations including confusion matrices, ROC curves, and feature importance plots
