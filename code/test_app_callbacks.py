from contextvars import copy_context
import mlflow

import pytest
import model3
import pickle
import numpy as np
import pandas as pd

submit = 1
def test_calculate_y_hardcode_1_plus_2_equal_3():
    output = model3.calculate_y_hardcode(1,2, submit)
    assert output == 3

def test_calculate_y_hardcode_2_plus_2_equal_4():
    output = model3.calculate_y_hardcode(2,2, submit)
    assert output == 4

# def test_model_input_shape():
#     filename = './models/Staging/values.pkl'
#     with open(filename, 'rb') as handle:
#         values = pickle.load(handle)
#     ohe = values['ohe']
#     scaler = values['scaler']
#     encoded_brand = list(ohe.transform([['Maruti']]).toarray()[0])
#     sample = np.array([[47.3, 2017, 1] + encoded_brand])
#     sample[:, 0: 2] = scaler.transform(sample[:, 0: 2])
#     sample = np.insert(sample, 0, 1, axis=1)
#     assert sample.shape == (1,35), f"Expecting the shape to be (1,35) but got {sample.shape=}" 

def test_model_output_shape():
    output = model3.calculate_model('Maruti',47.3,2017,1)
    assert output.shape == (1,), f"Expecting the shape to be (1,) but got {output.shape=}"