"""
Description:
    This module demonstrate on how the MLflow  tracking will work with machine learning model training.
    it train and predicts the Loan_status (Yes/No) based on several features like Gender,Married,Dependents,Education,
    Self_Employed,ApplicantIncome and Co-applicantIncome. This model will be trained on Logistic,Decision tree and
    random forrest algorithm and respective performance are being track with MLflow tracking features
Date :
    12th Nov,2024
Authore:
    Aditya Lakkad
Usage:
    - python loan_prediction.py
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
import mlflow
import os

# load the dataset
data = pd.read_csv('data/loan_eligibility.csv')

# imputing null values
data['Gender'].fillna(data['Gender'].mode()[0], inplace=True)
data['Married'].fillna(data['Married'].mode()[0], inplace=True)
data['Dependents'].fillna(data['Dependents'].mode()[0], inplace=True)
data['LoanAmount'].fillna(data['LoanAmount'].mean(), inplace=True)
data['Self_Employed'].fillna(data['Self_Employed'].mode()[0], inplace=True)
data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0], inplace=True)
data['Credit_History'].fillna(data['Credit_History'].mode()[0], inplace=True)

# removing outlier with standard IQR method
columns_to_filter = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
for col in columns_to_filter:
    q1 = data[col].quantile(0.25)
    q3 = data[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]

# feature creation
# adding applicant income and co-applicantinco asd total income
data['TotalIncome'] = data['ApplicantIncome'] + data['CoapplicantIncome']
# log transformation on totalincome and loanamount and laon_status(target)
data.TotalIncome = np.log(data.TotalIncome)
data.LoanAmount = np.log(data.LoanAmount)

# removing not required columns
data = data.drop(columns=['Loan_ID', 'ApplicantIncome', 'CoapplicantIncome'], axis=1)

# label encoding categorical data
for col in ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

# features and target
x = data.drop(columns=['Loan_Status'])
y = data.Loan_Status

# splitting data into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

# training Random forrest
rf = RandomForestClassifier(random_state=42)
# parameters for grid search
param_grid_rf = {
    'n_estimators': [200, 400, 700],
    'max_depth': [10, 20, 30],
    'criterion': ['gini', 'entropy'],
    'max_leaf_nodes': [50, 100]
}
grid_rf = GridSearchCV(
    estimator=rf,
    param_grid=param_grid_rf,
    cv=5,
    n_jobs=1,
    scoring='accuracy',
    verbose=0
)
model_rf = grid_rf.fit(x_train, y_train)

# Training Logistic regression
lr = LogisticRegression(random_state=42)
# parameters for grid search
param_grid_lr = {
    'C': [100, 10, 1.0 ,0.1, 0.01],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}
grid_lr = GridSearchCV(
    estimator=lr,
    param_grid=param_grid_lr,
    cv=5,
    n_jobs=1,
    scoring='accuracy',
    verbose=0
)
model_lr = grid_lr.fit(x_train, y_train)

# Training Decision Tree
dt = DecisionTreeClassifier(random_state=42)
# parameters for grid search
param_grid_dt = {
    'max_depth': [3, 5, 7, 9, 11, 13],
    'criterion': ['gini', 'entropy']
}
grid_dt = GridSearchCV(
    estimator=dt,
    param_grid=param_grid_dt,
    cv=5,
    n_jobs=1,
    scoring='accuracy',
    verbose=0
)
model_dt = grid_dt.fit(x_train, y_train)

def eval_metrics(y,pred):
    # accuracy
    accuracy = metrics.accuracy_score(y, pred)
    f1 = metrics.f1_score(y, pred, pos_label=1)

    # auc curve
    fpr, tpr, _ = metrics.roc_curve(y, pred)
    auc = metrics.auc(fpr, tpr)
    # plotting ROC curve
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='blue', label='ROC curve area = %0.2f' %auc)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    # Save plot
    os.makedirs('plots',exist_ok=True)
    plt.savefig('plots/ROC_Curve.png')
    plt.close()
    return (accuracy, f1, auc)

def mlflow_logging(model, x, y, name):

    with mlflow.start_run():

        mlflow.set_experiment('Loan-Prediction')

        # Setting uup runid as tag
        run_id = mlflow.active_run().info.run_id
        mlflow.set_tag('run_id', run_id)

        # predicting and evaluating metrics
        pred = model.predict(x)
        (accuracy, f1, auc) = eval_metrics(y, pred)
        # logging parameteres
        mlflow.log_params(model.best_params_)
        # log the metrics
        mlflow.log_metric('Mean CV score', model.best_score_)
        mlflow.log_metric('Accuracy', accuracy)
        mlflow.log_metric('F-1 Score', f1)
        mlflow.log_metric('AUC', auc)

        # logging AUC curve as a artifact
        mlflow.log_artifact('plots/ROC_Curve.png')
        mlflow.sklearn.log_model(model, name)


# logging Random forrest
mlflow.set_tracking_uri('http://localhost:5000')
# mlflow.create_experiment('Loan-Prediction')
mlflow_logging(model_rf, x_test, y_test, 'RandomForrestClassifier')
mlflow_logging(model_lr, x_test, y_test, 'LogisticRegression')
mlflow_logging(model_dt, x_test, y_test, 'DecisionTree')
