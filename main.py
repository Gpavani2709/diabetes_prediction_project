from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return "✅ Diabetes Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    # Extract values in the correct order
    features = [
        data['Pregnancies'],
        data['Glucose'],
        data['BloodPressure'],
        data['SkinThickness'],
        data['Insulin'],
        data['BMI'],
        data['DiabetesPedigreeFunction'],
        data['Age']
    ]
    
    input_array = np.array([features])
    prediction = model.predict(input_array)[0]
    
    result = "Diabetic" if prediction == 1 else "Not Diabetic"
    
    return jsonify({
        "prediction": int(prediction),
        "result": result
    })

if __name__ == '__main__':
    app.run(debug=True)
