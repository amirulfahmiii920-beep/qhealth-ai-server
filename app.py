from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import traceback
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)

# =======================================================
# LOAD BOTH AI MODELS
# =======================================================
try:
    model_screening = joblib.load('model_q1_screening.joblib')
    print("AI Model (Screening) loaded successfully!")
except Exception as e:
    print(f"Error loading screening model: {e}")
    model_screening = None

try:
    model_consultation = joblib.load('model_q2_doctor.joblib')
    print("AI Model (Consultation) loaded successfully!")
except Exception as e:
    print(f"Error loading consultation model: {e}")
    model_consultation = None

# Helper function to convert day number from Flutter to Day Name
def get_day_name(day_number):
    days = {
        1: 'Monday',
        2: 'Tuesday',
        3: 'Wednesday',
        4: 'Thursday',
        5: 'Friday',
        6: 'Saturday',
        7: 'Sunday',
        0: 'Sunday'
    }
    return days.get(day_number, 'Monday')

@app.route('/', methods=['GET'])
def home():
    return "Hospital AI Server is currently running with Dual Models!"

# =======================================================
# ROUTE 1: AI FOR SCREENING
# =======================================================
@app.route('/predict_ewt', methods=['POST'])
def predict_ewt():
    try:
        data = request.json
        print("\n--- REQUEST BY FLUTTER (SCREENING) ---")
        
        raw_hour = data.get('Entry_Hour', 8)
        raw_day = data.get('Entry_Day', 1) 
        
        
        raw_patient_type = data.get('Patient Type', 1)
        
        formatted_hour = float(raw_hour)
        formatted_day = get_day_name(raw_day)

        print(f"Data extracted: Hour={formatted_hour}, Day={formatted_day}, Patient Type={raw_patient_type}")
        
        
        features = pd.DataFrame([{
            'Entry_Hour': formatted_hour,
            'Entry_Day': formatted_day,
            'Patient Type': raw_patient_type
        }])
        
        print("AI Calculating Screening Time...")
        prediction = model_screening.predict(features)
        
        estimated_time = int(round(prediction[0]))
        print(f"AI prediction: {estimated_time} minutes")
        print("--------------------------------------\n")
        
        return jsonify({
            'success': True,
            'estimated_time_mins': estimated_time
        })

    except Exception as e:
        print("\n============= ERROR AI SCREENING =============")
        print(f"Cause: {str(e)}")
        traceback.print_exc()
        print("============================================\n")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# =======================================================
# ROUTE 2: AI FOR DOCTOR CONSULTATION
# =======================================================
@app.route('/predict_consult_ewt', methods=['POST'])
def predict_consult_ewt():
    try:
        data = request.json
        print("\n--- REQUEST BY FLUTTER (CONSULTATION) ---")
        
        # Extract Hour sent by Flutter
        raw_hour = data.get('Entry_Hour', 8.0)
        formatted_hour = float(raw_hour)
        
        # FIX ZON MASA MALAYSIA (Server Render guna masa UTC)
        now = datetime.utcnow() + timedelta(hours=8)
        current_day_name = now.strftime('%A')

        print(f"Data received: Hour={formatted_hour}, Day={current_day_name}")
        
        # TUKAR 'OUTPATIENT' KEPADA NOMBOR 1 (Supaya padan dengan format model AI)
        features = pd.DataFrame([{
            'Screening_End_Hour': formatted_hour,
            'Screening_End_Day': current_day_name,
            'Doctor Type': 'ANCHOR',      
            'Patient Type': 1  
        }])
        
        print("AI Calculating Consultation Time...")
        prediction = model_consultation.predict(features)
        
        estimated_time = int(round(prediction[0]))
        print(f"AI prediction: {estimated_time} minutes")
        print("--------------------------------------\n")
        
        return jsonify({
            'success': True,
            'estimated_time_mins': estimated_time
        })

    except Exception as e:
        print("\n============= ERROR AI CONSULTATION =============")
        print(f"Cause: {str(e)}")
        traceback.print_exc()
        print("============================================\n")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
