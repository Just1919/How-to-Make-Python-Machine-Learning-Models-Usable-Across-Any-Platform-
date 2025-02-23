import pickle
import pandas as pd
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Load the saved model
model = pickle.load(open('/Users/macbookp/Documents/Project/model.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if the request contains JSON
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request must contain JSON data"}), 400

        # Extract values from JSON
        age = data.get('age')
        sex = data.get('sex', '').lower()
        pclass = data.get('pclass')

        # Validate input
        if age is None or sex not in ['male', 'female'] or pclass not in [1, 2, 3]:
            return jsonify({"error": "Invalid input. Provide age (number), sex ('male' or 'female'), and pclass (1, 2, or 3)"}), 400
        
        age = float(age)
        pclass = int(pclass)

        # Convert categorical values to one-hot encoding
        sex_female = 1 if sex == 'female' else 0
        sex_male = 1 if sex == 'male' else 0
        pclass_1 = 1 if pclass == 1 else 0
        pclass_2 = 1 if pclass == 2 else 0
        pclass_3 = 1 if pclass == 3 else 0

        # Create DataFrame for prediction
        input_data = pd.DataFrame({
            'Age': [age],
            'Sex_female': [sex_female],
            'Sex_male': [sex_male],
            'Pclass_1': [pclass_1],
            'Pclass_2': [pclass_2],
            'Pclass_3': [pclass_3]
        })

        # Get prediction and probability
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]

        # Get dominant class and its probability
        dominant_class = int(prediction)
        probability = float(probabilities[dominant_class])

        # Map the prediction to "survived" or "not survived"
        survival_status = "survived" if dominant_class == 1 else "not survived"

        # Return result in JSON format
        return jsonify({
            "prediction": survival_status,
            "probability": round(probability, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
