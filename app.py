from flask import Flask, request, jsonify, render_template
import pickle  # or use joblib if your model is saved with joblib
import numpy as np

app = Flask(__name__)

# Load the model from the pickle file
try:
    with open('model.pkl', 'rb') as file:  # Change to 'model.joblib' if using joblib
        model = pickle.load(file)  # or use joblib.load('model.joblib')
except Exception as e:
    print("Error loading model:", e)

# Route to render the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Define a route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = data['features']
        features_array = np.array(features).reshape(1, -1)  # Reshape for a single sample
        prediction = model.predict(features_array)
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
