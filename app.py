from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
# from keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ============================================================================
# HELPER FUNCTIONS (MOVE THIS BEFORE LOADING MODELS)
# ============================================================================

class FeatureEngineer:
    def __init__(self, symptom_severity_df):
        self.severity_dict = dict(zip(
            symptom_severity_df['Symptom'],
            symptom_severity_df['weight']
        ))
    
    def create_severity_features(self, X):
        X_enhanced = X.copy()
        
        # Total severity
        severity_scores = []
        for idx, row in X.iterrows():
            total_severity = sum(
                self.severity_dict.get(col, 0) * row[col]
                for col in X.columns
            )
            severity_scores.append(total_severity)
        X_enhanced['total_severity'] = severity_scores
        
        # Average severity
        avg_severity = []
        for idx, row in X.iterrows():
            present_symptoms = [col for col in X.columns if row[col] == 1]
            if present_symptoms:
                avg = np.mean([self.severity_dict.get(sym, 0) for sym in present_symptoms])
            else:
                avg = 0
            avg_severity.append(avg)
        X_enhanced['avg_severity'] = avg_severity
        
        # Max severity
        max_severity = []
        for idx, row in X.iterrows():
            present_symptoms = [col for col in X.columns if row[col] == 1]
            if present_symptoms:
                max_sev = max([self.severity_dict.get(sym, 0) for sym in present_symptoms])
            else:
                max_sev = 0
            max_severity.append(max_sev)
        X_enhanced['max_severity'] = max_severity
        
        # Number of symptoms
        X_enhanced['num_symptoms'] = X.sum(axis=1)
        
        return X_enhanced
    
    def create_interaction_features(self, X, top_symptoms):
        X_enhanced = X.copy()
        
        for i in range(len(top_symptoms)):
            for j in range(i + 1, len(top_symptoms)):
                sym1 = top_symptoms[i]
                sym2 = top_symptoms[j]
                
                if sym1 in X.columns and sym2 in X.columns:
                    interaction_name = f"{sym1[:15]}_{sym2[:15]}_interact"
                    X_enhanced[interaction_name] = X[sym1] * X[sym2]
        
        return X_enhanced

# ============================================================================
# LOAD MODELS AND PREPROCESSING OBJECTS
# ============================================================================

print("Loading models and preprocessing objects...")

# Load ensemble models
with open(r'C:\Users\hp\Desktop\himans_avi\Models\ensemble_models.pkl', 'rb') as f:
    ensemble_models = pickle.load(f)

# Load neural network
# ann_model = load_model(r'C:\Users\hp\Desktop\himans_avi\Models\neural_network_model.h5')

# Load preprocessing objects
with open(r'C:\Users\hp\Desktop\himans_avi\Models\preprocessing_objects.pkl', 'rb') as f:
    preprocessing = pickle.load(f)

# Load symptom dictionary
with open(r'C:\Users\hp\Desktop\himans_avi\Models\symptoms_dict.pkl', 'rb') as f:
    symptoms_dict = pickle.load(f)

# Load disease list
with open(r'C:\Users\hp\Desktop\himans_avi\Models\diseases_list.pkl', 'rb') as f:
    diseases_list = pickle.load(f)

# Load medical data
description_df = pd.read_csv('description.csv')
medications_df = pd.read_csv('medications.csv')
precautions_df = pd.read_csv('precautions_df.csv')
diets_df = pd.read_csv('diets.csv')
workout_df = pd.read_csv('workout_df.csv')
symptom_severity_df = pd.read_csv('Symptom-severity.csv')

print("✓ All models and data loaded successfully!")

# Initialize feature engineer
feature_engineer = FeatureEngineer(symptom_severity_df)
top_symptoms = ['fatigue', 'vomiting', 'high_fever', 'loss_of_appetite', 
                'nausea', 'headache', 'abdominal_pain', 'yellowish_skin']

label_encoder = preprocessing['label_encoder']
scaler = preprocessing['scaler']
selected_features = preprocessing['selected_features']

# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def preprocess_symptoms(symptoms):
    """Convert symptom list to feature vector"""
    # Create base vector
    base_vector = np.zeros(len(symptoms_dict))
    
    valid_symptoms = []
    for symptom in symptoms:
        symptom_clean = symptom.lower().strip().replace(' ', '_')
        if symptom_clean in symptoms_dict:
            base_vector[symptoms_dict[symptom_clean]] = 1
            valid_symptoms.append(symptom_clean)
    
    # Convert to dataframe
    symptom_columns = list(symptoms_dict.keys())
    base_df = pd.DataFrame([base_vector], columns=symptom_columns)
    
    # Apply feature engineering
    enhanced_df = feature_engineer.create_severity_features(base_df)
    enhanced_df = feature_engineer.create_interaction_features(enhanced_df, top_symptoms)
    
    # Select features
    available_features = [f for f in selected_features if f in enhanced_df.columns]
    
    # Add missing features
    for feat in selected_features:
        if feat not in enhanced_df.columns:
            enhanced_df[feat] = 0
    
    feature_vector = enhanced_df[selected_features].values[0]
    
    return feature_vector, valid_symptoms

def get_medical_info(disease):
    """Get medical recommendations for disease"""
    info = {
        'description': 'No description available.',
        'precautions': [],
        'medications': [],
        'diet': [],
        'workout': []
    }
    
    # Description
    desc_match = description_df[description_df['Disease'] == disease]['Description']
    if not desc_match.empty:
        info['description'] = desc_match.values[0]
    
    # Precautions
    prec_match = precautions_df[precautions_df['Disease'] == disease]
    if not prec_match.empty:
        prec_cols = ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']
        info['precautions'] = [p for p in prec_match[prec_cols].values[0] if pd.notna(p)]
    
    # Medications
    med_match = medications_df[medications_df['Disease'] == disease]['Medication']
    if not med_match.empty:
        info['medications'] = eval(med_match.values[0]) if isinstance(med_match.values[0], str) else [med_match.values[0]]
    
    # Diet
    diet_match = diets_df[diets_df['Disease'] == disease]['Diet']
    if not diet_match.empty:
        info['diet'] = eval(diet_match.values[0]) if isinstance(diet_match.values[0], str) else [diet_match.values[0]]
    
    # Workout
    workout_match = workout_df[workout_df['disease'] == disease]['workout']
    if not workout_match.empty:
        info['workout'] = workout_match.values.tolist()[:5]
    
    return info

def predict_disease(symptoms, model_name='Stacking Ensemble'):
    """Make prediction"""
    try:
        # Preprocess
        feature_vector, valid_symptoms = preprocess_symptoms(symptoms)
        
        if len(valid_symptoms) == 0:
            return {'error': 'No valid symptoms provided', 'success': False}
        
        # Get model
        if model_name == 'Neural Network':
            # model = ann_model
            # feature_vector_scaled = scaler.transform([feature_vector])
            # probabilities = model.predict(feature_vector_scaled, verbose=0)[0]
            return {'error': 'Neural Network model is currently disabled', 'success': False}
        else:
            model = ensemble_models[model_name.lower().replace(' ', '_')]
            probabilities = model.predict_proba([feature_vector])[0]
        
        # Get top 3 predictions
        top_3_idx = np.argsort(probabilities)[::-1][:3]
        
        predictions = []
        for idx in top_3_idx:
            disease = label_encoder.classes_[idx]
            confidence = float(probabilities[idx])
            
            predictions.append({
                'disease': disease,
                'confidence': confidence,
                'medical_info': get_medical_info(disease)
            })
        
        return {
            'success': True,
            'model_used': model_name,
            'symptoms_analyzed': valid_symptoms,
            'predictions': predictions,
            'primary_diagnosis': predictions[0]['disease'],
            'primary_confidence': predictions[0]['confidence']
        }
        
    except Exception as e:
        return {'error': str(e), 'success': False}

# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def home():
    """Home page"""
    return render_template('index.html', symptoms=list(symptoms_dict.keys()))

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        data = request.get_json()
        symptoms = data.get('symptoms', [])
        model_name = data.get('model', 'Stacking Ensemble')
        
        if not symptoms:
            return jsonify({'error': 'No symptoms provided', 'success': False})
        
        result = predict_disease(symptoms, model_name)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False})

@app.route('/symptoms')
def get_symptoms():
    """Get all available symptoms"""
    return jsonify({
        'symptoms': sorted(list(symptoms_dict.keys())),
        'count': len(symptoms_dict)
    })

@app.route('/models')
def get_models():
    """Get available models"""
    return jsonify({
        'models': [
            'Random Forest',
            'XGBoost',
            'Gradient Boosting',
            'Voting Ensemble',
            'Stacking Ensemble',
            # 'Neural Network'  # Commented out as ANN is disabled
        ]
    })

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)