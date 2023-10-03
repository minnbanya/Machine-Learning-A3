def calculate_y_hardcode(x_1, x_2, submit):
    print(x_1)
    print(x_2)
    print(submit)
    return x_1 + x_2

def calculate_y_model(x_1,x_2,x_3,x_4, submit):
    pred = calculate_model(x_1,x_2,x_3,x_4)
    return f" model said: {pred=}" # type:ignore

def calculate_model(x_1,x_2,x_3,x_4):
    from utils import load_mlflow
    import pandas as pd
    import numpy as np
    import mlflow
    import pickle
    mlflow.set_tracking_uri("https://mlflow.cs.ait.ac.th/")
    model_name = "st124145-a3-model"
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/Staging")
    filename = './models/Staging/values.pkl'
    with open(filename, 'rb') as handle:
        values = pickle.load(handle)
    ohe = values['ohe']
    poly = values['poly']
    scaler = values['scaler']
    encoded_brand = list(ohe.transform([[x_1]]).toarray()[0])
    sample = np.array([[x_2,x_3,x_4] + encoded_brand])
    sample[:, 0: 2] = scaler.transform(sample[:, 0: 2])
    sample = np.insert(sample, 0, 1, axis=1)
    sample_poly = poly.transform(sample)
    pred = model.predict(sample_poly) # type:ignore
    return pred