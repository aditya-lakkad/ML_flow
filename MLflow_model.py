"""
Description:
    this scripts fetches the already created model (check loan_eligibility.py) , deploye it as a REST API and get the
    results through it
Date :
    13th Nov,2024
Author:
    Aditya Lakkad
Usage:
    -
"""
import mlflow
mlflow.set_tracking_uri('http://localhost:5000')
logged_model = 'runs:/2b42b068fecc4e178a4007f90137f561/LogisticRegression'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd
data = [[1.0, 0.0, 0.0, 0.0, 0.0, 4.98, 360.0, 1.0, 2.0, 8.60]]  # sample input data
print(loaded_model.predict(pd.DataFrame(data)))