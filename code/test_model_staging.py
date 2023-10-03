"""
The idea of this file is to test the model at Staging.
If it passes the test, we will automtically move the Model to Production.

This file will test the model in Staging.
If the model in Staging is tested, on the production level, 
"""
from utils import load_mlflow,load_values
import numpy as np
import pandas as pd
import pytest
import mlflow
# I don't need to set mlflow.set_tracking_uri()
# because I set it in the environment of this container during compose up.
# With this, people who has my image won't know the link to the mlflow server.
stage = "Staging"
values = load_values()
ohe = values['ohe']
def test_load_model():
    mlflow.set_tracking_uri("https://mlflow.cs.ait.ac.th/")
    model_name = "st124145-a3-model"
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/Staging")
    assert model

@pytest.mark.depends(on=['test_load_model'])
def test_model_input():
    mlflow.set_tracking_uri("https://mlflow.cs.ait.ac.th/")
    model_name = "st124145-a3-model"
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/Staging")
    encoded_brand = list(ohe.transform([['Maruti']]).toarray()[0])
    X = np.array([1,2,3]+encoded_brand).reshape(-1,4)
    X = pd.DataFrame(X, columns=['x1', 'x2','x3']+ohe.categories_)
    pred = model.predict(X) # type:ignore
    assert pred

@pytest.mark.depends(on=['test_model_input'])
def test_model_output():
    mlflow.set_tracking_uri("https://mlflow.cs.ait.ac.th/")
    model_name = "st124145-a3-model"
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/Staging")
    encoded_brand = list(ohe.transform([['Maruti']]).toarray()[0])
    X = np.array([1,2,3]+encoded_brand).reshape(-1,4)
    X = pd.DataFrame(X, columns=['x1', 'x2','x3']+ohe.categories_)
    pred = model.predict(X) # type:ignore
    assert pred.shape == (1,1), f"{pred.shape=}"

@pytest.mark.depends(on=['test_load_model'])
def test_model_coeff():
    mlflow.set_tracking_uri("https://mlflow.cs.ait.ac.th/")
    model_name = "st124145-a3-model"
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/Staging")
    assert model._coef().shape == (8434,4), f"{model._coef().shape=}" # type:ignore