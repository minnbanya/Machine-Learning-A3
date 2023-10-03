def calculate_y_hardcode(x_1, x_2, submit):
    print(x_1)
    print(x_2)
    print(submit)
    return x_1 + x_2

def calculate_y_model(x_1, x_2, submit):
    pred = calculate_model(x_1,x_2)
    coef = get_coeff()
    return f" model said: {pred=} {coef=}" # type:ignore

def get_coeff():
    from utils import load_mlflow
    model = load_mlflow(stage="Production")
    return model.coef_ # type:ignore

def calculate_model(x_1,x_2):
    from utils import load_mlflow
    import pandas as pd
    import numpy as np
    model = load_mlflow(stage="Production")
    X = np.array([x_1,x_2]).reshape(-1,2)
    X = pd.DataFrame(X, columns=['x1', 'x2']) 
    pred = model.predict(X) # type:ignore
    return pred