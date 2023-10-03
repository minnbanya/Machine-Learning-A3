# Importing libraries
from flask import Flask, render_template, request, flash
import pickle
import numpy as np
from linear_regression import LinearRegression, NormalPenalty, LassoPenalty, RidgePenalty, ElasticPenalty, Normal, Lasso, Ridge, ElasticNet
# from logistic_regression import LogisticRegression, NormalPenalty, LassoPenalty, RidgePenalty, ElasticPenalty, Normal, Lasso, Ridge, ElasticNet
import mlflow.pyfunc


app = Flask(__name__)

# Setting a secret key to use flash
app.secret_key = 'ml2023'


# Importing the old model
filename1 = 'models/car_price_old.model'
loaded_data1 = pickle.load(open(filename1, 'rb'))

# Separating the values in the old model file into variables for easy access
model_old = loaded_data1['model']
scaler_old = loaded_data1['scaler']
name_map_old = loaded_data1['name_map']
engine_default_old = loaded_data1['engine_default']
mileage_default_old = loaded_data1['mileage_default']

# Importing the new model
filename2 = 'models/car_price_new.model'
loaded_data2 = pickle.load(open(filename2, 'rb'))

# Separating the values in the new model file into variables for easy access
model_new = loaded_data2['model']
scaler_new = loaded_data2['scaler']
name_map_new = loaded_data2['name_map']
engine_default_new = loaded_data2['engine_default']
mileage_default_new = loaded_data2['mileage_default']

# Loading the default values and classification model
filename = 'models/Staging/values.pkl'
values_class = pickle.load(open(filename,'rb'))

mlflow.set_tracking_uri("https://mlflow.cs.ait.ac.th/")
model_name = "st124145-a3-model"
model_class = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/Staging")
# Setting app to run on Flask

k_range = values_class['k_range']
scaler_class = values_class['scaler']
ohe = values_class['ohe']
poly = values_class['poly']
fuel_class = values_class['fuel_default']
year_class = values_class['year_default']
max_class = values_class['max_default']

# The home page containing a link to the prediction page
@app.route('/')
def index():
    return render_template('index.html')

# The prediction page
@app.route('/predict_old')
def predict_old():
    return render_template('predict_old.html')

# The route to calculate the prediction result but not accessed by users
@app.route('/process-data_old', methods = ['POST'])
def process_data_old():
    if request.method == 'POST':
        # Getting the values required for prediction
        brand_name = request.form.get('name')
        name = name_map_old.get(brand_name,'32')
        engine = request.form.get('engine', engine_default_old)
        mileage = request.form.get('mileage', mileage_default_old)

        # Convert engine and mileage to float only if they are not empty strings
        if engine:
            engine = float(engine)
        else:
            engine = engine_default_old  # Set a default value if engine is empty

        if mileage:
            mileage = float(mileage)
        else:
            mileage = mileage_default_old # Set a default value if mileage is empty

        # Calling the prediction function, coverting the result to int for user experience and then to string
        # to display on the website
        result = str(int(prediction_old(name,engine,mileage)[0]))

        return result

# Prediction function to predit car price
def prediction_old(name,engine,mileage):
    # Put the user input into an array
    sample = np.array([[name,engine,mileage]])

    # Scale the input data using the trained scaler
    sample_scaled = scaler_old.transform(sample)

    # Predict the car price using the trained model
    result = np.exp(model_old.predict(sample_scaled))

    return result

# The prediction page
@app.route('/predict_new')
def predict_new():
    flash('Hey, you can use the new model in the same way you use the old model. This model is trained from scratch so that it is better suited to predict the price of your car!', 'success')
    return render_template('predict_new.html')

# The route to calculate the prediction result but not accessed by users
@app.route('/process-data_new', methods = ['POST'])
def process_data_new():
    if request.method == 'POST':
        # Getting the values required for prediction
        brand_name = request.form.get('name')
        name = name_map_new.get(brand_name,'32')
        engine = request.form.get('engine', engine_default_new)
        mileage = request.form.get('mileage', mileage_default_new)

        # Convert engine and mileage to float only if they are not empty strings
        if engine:
            engine = float(engine)
        else:
            engine = engine_default_new  # Set a default value if engine is empty

        if mileage:
            mileage = float(mileage)
        else:
            mileage = mileage_default_new # Set a default value if mileage is empty

        # Calling the prediction function, coverting the result to int for user experience and then to string
        # to display on the website
        result = str(int(prediction_new(name,engine,mileage)[0]))

        return result

# Prediction function to predit car price
def prediction_new(name,engine,mileage):
    # Put the user input into an array
    sample = np.array([[name,engine,mileage]])

    # Scale the input data using the trained scaler and add intercepts
    sample_scaled = scaler_new.transform(sample)
    intercept = np.ones((sample_scaled.shape[0], 1))
    sample_scaled   = np.concatenate((intercept, sample_scaled), axis=1)

    # Predict the car price using the trained model
    result = np.exp(model_new.predict(sample_scaled))

    return result

@app.route('/predict_class')
def predict_class():
    return render_template('predict_class.html')

# The route to calculate the prediction result but not accessed by users
@app.route('/process-data_class', methods = ['POST'])
def process_data_class():
    if request.method == 'POST':
        # Getting the values required for prediction
        brand_name = request.form.get('name')
        name = list(ohe.transform([[brand_name]]).toarray()[0])
        fuel = request.form.get('fuel', fuel_class)
        year = request.form.get('year', year_class)
        max = request.form.get('max', max_class)

        # Convert fuel and year and max to float only if they are not empty strings
        if fuel:
            fuel = float(fuel)
        else:
            fuel = fuel_class  # Set a default value if fuel is empty

        if year:
            year = float(year)
        else:
            year = year_class # Set a default value if year is empty

        if max:
            max = float(max)
        else:
            max = max_class # Set a default value if max is empty

        # Calling the prediction function, coverting the result to int for user experience and then to string
        # to display on the website
        result = prediction_class(name,max,year,fuel)

        return result

# Prediction function to predit car price
def prediction_class(name,max_power,year,fuel):
    sample = np.array([[max_power,year,fuel]+name])

    # Scale the input data using the trained scaler
    sample[:, 0: 2] = scaler_class.transform(sample[:, 0: 2])
    sample = np.insert(sample, 0, 1, axis=1)
    sample_poly = poly.transform(sample)
    # Predict the car price using the trained model
    result = model_class.predict(sample_poly)

    return k_range[result[0]]


port_number = 8000

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=port_number)