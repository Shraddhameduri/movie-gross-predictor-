
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import numpy as np
from joblib import dump, load
import os
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # Importing matplotlib.pyplot as plt
import random
app = Flask(__name__)
CORS(app)

model = None

@app.before_first_request
def initialize_model():
    global model
    print("Initializing model...")
    model_file = 'model.joblib'
    if os.path.exists(model_file):
        # Load the model from file
        print("Loading existing model...")
        model = load(model_file)
    else:
        # Train a new model and save it to the file
        print("Model file not found. Training a new model...")
        model = train_model()
        dump(model, model_file)



@app.route('/api/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not trained yet.'}), 400
    
    data = request.get_json()
    required_fields = ['year', 'score', 'votes', 'runtime', 'rating', 'genre', 'director', 'writer', 'star', 'country', 'company']
    for field in required_fields:
        if field not in data or data[field] == '':
            return jsonify({'error': f'Missing or invalid value for field: {field}'}), 400
    input_data = pd.DataFrame([data])
    predicted_gross_log = model.predict(input_data)
    print(predicted_gross_log)
    response = {
        'predicted_gross': float(predicted_gross_log[0])  # Convert the predicted value to float
    } 
    
    # Return the response as JSON
    return jsonify(response), 200


@app.route('/api/plot-predicted-vs-actual', methods=['POST'])
def plot_predicted_vs_actual():
    data_path = './cleaned_movies.xlsx'
    df = pd.read_excel(data_path)
    features = ['year', 'score', 'votes', 'runtime', 'rating', 'genre', 'director', 'writer', 'star', 'country', 'company']
    target = 'gross_log'
    _, X_test, _, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)
    y_pred_log = model.predict(X_test)
    y_pred = np.exp(y_pred_log)
    y_actual = np.exp(y_test)
    y_min, y_max = np.percentile(y_actual, [10, 90])
    new_data = {}
    for feature in features:
        value = request.form.get(feature)
        if value is None:
            return jsonify({'error': f'Missing parameter: {feature}'}), 400
        try:
            if feature in ['year', 'votes', 'runtime']:
                value = int(value)
            else:
                value = str(value)
        except ValueError:
            return jsonify({'error': f'Invalid value for {feature}'}), 400
        
        new_data[feature] = value
    new_input_data = pd.DataFrame([new_data])
    
    new_pred_log = model.predict(new_input_data)
    new_pred = np.exp(new_pred_log)[0]
    plt.figure(figsize=(8, 6))
    plt.scatter(y_actual, y_pred, alpha=0.5, label='Actual vs. Predicted')
    plt.scatter(new_pred, new_pred, color='red', label='Newly predicted point')
    plt.plot([min(y_actual), max(y_actual)], [min(y_actual), max(y_actual)], color='red', linestyle='--', label='Prediction Line')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs. Predicted Values')
    plt.legend()
    plt.ylim(y_min, y_max)
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    return send_file(buffer, mimetype='image/png')



@app.route('/api/random-data-point', methods=['GET'])
def get_random_data_point():
    data_path = './cleaned_movies.xlsx'
    df = pd.read_excel(data_path)
    random_index = random.randint(0, len(df) - 1)
    random_data_point = df.iloc[random_index]
    data_point_dict = {
        'name': random_data_point.get('name'),
        'rating': random_data_point.get('rating'),
        'genre': random_data_point.get('genre'),
        'year': str(random_data_point.get('year')),
        'released': random_data_point.get('released'),
        'score': str(random_data_point.get('score')),
        'votes': str(random_data_point.get('votes')),
        'director': random_data_point.get('director'),
        'writer': random_data_point.get('writer'),
        'star': random_data_point.get('star'),
        'country': random_data_point.get('country'),
        'company': random_data_point.get('company'),
        'runtime': str(random_data_point.get('runtime')),
        'gross_log': random_data_point.get('gross_log')  
    }
    
    return jsonify(data_point_dict)

if __name__ == '__main__':
    app.run(debug=True)

        
def train_model():
    data_path = './cleaned_movies.xlsx'
    df = pd.read_excel(data_path)
    features = ['year', 'score', 'votes', 'runtime', 'rating', 'genre', 'director', 'writer', 'star', 'country', 'company']
    target = 'gross_log'
    for feature in features:
        if feature not in df.columns:
            raise ValueError(f"The feature '{feature}' is not a column in the dataframe.")
    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)
    
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['rating', 'genre', 'director', 'writer', 'star', 'country', 'company']),
        ('num', StandardScaler(), ['year', 'score', 'votes', 'runtime'])
    ])
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('poly_features', PolynomialFeatures(degree=2, include_bias=False)),
        ('regressor', LinearRegression())
    ])
    
    model_pipeline.fit(X_train, y_train)
    
    print("Model training completed.")
    return model_pipeline

if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(host="0.0.0.0", port=8000, debug=True)