from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the 7-feature model and imputer
model = joblib.load('xgboost_model_7.pkl')
imputer = joblib.load('imputer_7.pkl')

@app.route('/')
def home():
    # Pass inputs=None so fields initialize completely blank
    return render_template('index.html', inputs=None)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # 1. Extract values
        pregnancies = float(request.form['pregnancies'])
        glucose = float(request.form['glucose'])
        blood_pressure = float(request.form['blood_pressure'])
        skin_thickness = float(request.form['skin_thickness'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        age = float(request.form['age'])
        
        # 2. Process inputs for prediction
        features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, age]])
        clean_features = imputer.transform(features)
        
        probabilities = model.predict_proba(clean_features)
        diabetes_probability = probabilities[0][1]
        
        # 3. Determine tier and messaging
        if diabetes_probability >= 0.70:
            result_title = "⚠️ High Risk Profile Detected"
            result_desc = f"Based on your metrics, there is a high correlation with diabetes indicators. We strongly suggest consulting a medical professional for a comprehensive evaluation."
            color = "#d9534f" 
            
        elif 0.30 <= diabetes_probability < 0.70:
            result_title = "🟡 Medium Risk Profile (Borderline / Prediabetes)"
            result_desc = f"Your indicators fall into a moderate risk range. This often points toward prediabetes, meaning your glucose levels are elevated but manageable. Small, proactive changes in lifestyle and diet can effectively reverse this direction."
            color = "#f0ad4e" 
            
        else:
            result_title = "✅ Low Risk Profile Detected"
            result_desc = f"Great news! Your indicators place you in a low risk tier. Continue practicing your healthy habits to protect and maintain your overall wellness."
            color = "#2b8a3e"
            
        # Returning request.form as 'inputs' preserves the text box data on submit
        return render_template('index.html', 
                               prediction_title=result_title, 
                               prediction_desc=result_desc, 
                               result_color=color,
                               inputs=request.form)

if __name__ == '__main__':
    app.run(debug=True)