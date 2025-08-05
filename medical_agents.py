# medical_agents.py

import pandas as pd
import numpy as np
import joblib
from scipy.sparse import hstack
import json
import re
from typing import Dict, List
import warnings
import urllib.parse

# Machine Learning
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# Optional imports with error handling
try:
    import xgboost as xgb
    print("✅ XGBoost imported successfully")
except ImportError:
    print("⚠️ XGBoost not found - needed for heart disease model")

# Suppress warnings
warnings.filterwarnings('ignore')

# Your LoadedMedicalTriageAgent class (from your notebook)
class LoadedMedicalTriageAgent:
    def __init__(self, saved_components):
        self.symptom_classifier = saved_components['symptom_classifier']
        self.word_vec = saved_components['word_vectorizer']
        self.char_vec = saved_components['char_vectorizer']
        self.label_encoder = saved_components['label_encoder']
        self.specialist_mapping = saved_components.get('specialist_mapping', {})
        self.target_diseases = saved_components.get('target_diseases', {})
        self.medical_mappings = saved_components.get('medical_mappings', {})
    
    def preprocess_text(self, text: str) -> str:
        text = re.sub(r'[^\w\s\-\.]', ' ', text.lower())
        for k, v in self.medical_mappings.items():
            text = text.replace(k, v)
        return re.sub(r'\s+', ' ', text).strip()
    
    def classify(self, text: str):
        """Enhanced classify method with your training approach"""
        txt = self.preprocess_text(text)
        
        # Use the same feature engineering as your training
        X_word = self.word_vec.transform([txt])
        X_char = self.char_vec.transform([txt])
        
        # Keyword features
        keyword_bank = {
            'kw_diabetes': self.specialist_mapping.get('diabetes', {}).get('keywords', []),
            'kw_hypertension': self.specialist_mapping.get('hypertension', {}).get('keywords', []),
            'kw_asthma': self.specialist_mapping.get('bronchial_asthma', {}).get('keywords', []),
            'kw_dengue': self.specialist_mapping.get('dengue', {}).get('keywords', []),
            'kw_malaria': self.specialist_mapping.get('malaria', {}).get('keywords', []),
            'kw_common_cold': self.specialist_mapping.get('common_cold', {}).get('keywords', [])
        }
        
        X_kw = self._build_keyword_matrix(pd.Series([txt]), keyword_bank)
        X_full = hstack([X_word, X_char, X_kw])
        
        pred = self.symptom_classifier.predict(X_full)[0]
        probs = self.symptom_classifier.predict_proba(X_full)[0]
        disease = self.label_encoder.inverse_transform([pred])[0]
        conf = float(np.max(probs))
        
        # Apply rule-based overrides
        rule_result = self._apply_rules(txt, disease, conf)
        if rule_result:
            return rule_result
        
        act_map = {
            'diabetes': 'MODERATE – doctor in 1-2 days',
            'hypertension': 'MODERATE – monitor BP',
            'bronchial_asthma': 'URGENT – breathing care',
            'dengue': 'URGENT – hospital eval',
            'malaria': 'URGENT – antimalarial Rx',
            'common_cold': 'ROUTINE – self-care'
        }
        
        return {
            'predicted_disease': disease,
            'confidence': conf,
            'confidence_level': 'high' if conf >= 0.8 else 'medium' if conf >= 0.6 else 'low',
            'recommended_action': act_map.get(disease, 'ROUTINE – consult doctor'),
            'rule_override': False
        }
    
    def _build_keyword_matrix(self, text_series, keyword_bank):
        """Build keyword matrix for feature engineering"""
        from scipy.sparse import csr_matrix
        rows, cols, data = [], [], []
        for row_idx, text in enumerate(text_series):
            low = text.lower()
            for col_idx, (feat, kws) in enumerate(keyword_bank.items()):
                if kws and any(k in low for k in kws):
                    rows.append(row_idx)
                    cols.append(col_idx)
                    data.append(1)
        return csr_matrix((data, (rows, cols)), 
                         shape=(len(text_series), len(keyword_bank)))
    
    def _apply_rules(self, txt_low, disease, conf):
        """Apply rule-based overrides"""
        strong_patterns = {
            'bronchial_asthma': [
                r'\b(asthma|asthmatic)\b',
                r'\bwheez\w+\b',
                r'\bshort(ness)? of breath\b',
                r'\bbreathlessness\b',
                r'\bchest tightness\b'
            ],
            'malaria': [
                r'\bmalaria\b',
                r'\bplasmodium\b',
                r'\brigor?s?\b',
                r'\bcyclic fever\b',
                r'\bblood smear\b',
                r'\bfever(?:.*)(every|each)\s+(24|48|72)\s*hours\b'
            ],
            'common_cold': [
                r'\brunny nose\b',
                r'\bnasal congestion\b',
                r'\bsneez\w+\b',
                r'\bsore throat\b'
            ]
        }
        
        for rule_disease, patterns in strong_patterns.items():
            if any(re.search(pat, txt_low) for pat in patterns):
                act_map = {
                    'diabetes': 'MODERATE – doctor in 1-2 days',
                    'hypertension': 'MODERATE – monitor BP',
                    'bronchial_asthma': 'URGENT – breathing care',
                    'dengue': 'URGENT – hospital eval',
                    'malaria': 'URGENT – antimalarial Rx',
                    'common_cold': 'ROUTINE – self-care'
                }
                
                final_conf = max(conf, 0.85)
                return {
                    'predicted_disease': rule_disease,
                    'confidence': final_conf,
                    'confidence_level': 'high' if final_conf >= 0.8 else 'medium' if final_conf >= 0.6 else 'low',
                    'recommended_action': act_map.get(rule_disease, 'ROUTINE – consult doctor'),
                    'rule_override': True
                }
        return None

# Your SimpleManualHospitalFinder class (from your notebook)
class SimpleManualHospitalFinder:
    """Simple manual hospital finder - creates Google Maps links manually"""
    
    def __init__(self):
        print("✅ Simple Manual Hospital Finder initialized")
    
    def create_google_maps_search_link(self, location, search_type='hospital'):
        """Create a simple Google Maps search link manually"""
        # Create search query
        if search_type == 'diabetes':
            search_query = f"diabetes hospital endocrinology clinic near {location}"
        elif search_type == 'hypertension':
            search_query = f"cardiology hospital heart clinic near {location}"
        else:
            search_query = f"hospital near {location}"
        
        # Encode the search query for URL
        encoded_query = urllib.parse.quote(search_query)
        
        # Create Google Maps search URL
        google_maps_link = f"https://www.google.com/maps/search/{encoded_query}"
        
        return google_maps_link
    
    def find_hospitals_manual(self, location, condition_type='general'):
        """Return manual hospital recommendations with Google Maps links"""
        
        print(f"🔍 Creating manual Google Maps search for {condition_type} hospitals near {location}...")
        
        # Create Google Maps search link
        maps_link = self.create_google_maps_search_link(location, condition_type)
        
        # Create manual hospital recommendations
        hospitals = [
            {
                'name': f'Search: Hospitals near {location}',
                'search_type': f'{condition_type.title()} hospitals and clinics',
                'google_maps_link': maps_link,
                'instructions': 'Click the Google Maps link to find nearby hospitals',
                'emergency_contact': '108 (Ambulance) | 102 (Medical Emergency)',
                'manual_note': 'Use Google Maps to find exact locations, get directions, and see reviews'
            }
        ]
        
        return hospitals

# Your EnhancedMedicalSpecialistAgent class (from your notebook)
class EnhancedMedicalSpecialistAgent:
    """Complete medical consultation system with triage -> specialist -> consultation flow"""
    
    def __init__(self, triage_agent, specialist_models):
        self.triage_agent = triage_agent
        self.specialist_models = specialist_models
        
        # Initialize manual hospital finder
        self.hospital_finder = SimpleManualHospitalFinder()
        print("✅ Enhanced Medical Specialist Agent initialized successfully!")
    
    def predict_diabetes_risk(self, patient_data):
        """Predict diabetes risk using specialist model"""
        try:
            models = self.specialist_models['diabetes']
            
            # Encode categorical variables
            gender_encoded = models['gender_encoder'].transform([patient_data['gender']])[0]
            smoking_encoded = models['smoking_encoder'].transform([patient_data['smoking_history']])[0]
            
            # Prepare features (matching your training data structure)
            features = [
                gender_encoded,
                patient_data['age'],
                patient_data['hypertension'],
                patient_data['heart_disease'],
                smoking_encoded,
                patient_data['bmi'],
                patient_data['blood_glucose_level']
            ]
            
            # Make prediction
            prediction = models['model'].predict([features])[0]
            probabilities = models['model'].predict_proba([features])[0]
            
            confidence = float(np.max(probabilities))
            
            return {
                'prediction': int(prediction),
                'confidence': confidence,
                'risk_level': 'High' if prediction == 1 else 'Low',
                'probability_diabetes': float(probabilities[1]) if len(probabilities) > 1 else confidence
            }
        
        except Exception as e:
            print(f"❌ Diabetes prediction error: {e}")
            return None
    
    def predict_heart_disease_risk(self, patient_data):
        """Predict heart disease risk using specialist model"""
        try:
            models = self.specialist_models['heart']
            
            # Prepare features
            features = [[
                patient_data['age'], patient_data['sex'], patient_data['cp'],
                patient_data['trestbps'], patient_data['chol'], patient_data['fbs'],
                patient_data['restecg'], patient_data['thalach'], patient_data['exang'],
                patient_data['oldpeak'], patient_data['slope'], patient_data['ca'], patient_data['thal']
            ]]
            
            # Scale features
            features_scaled = models['scaler'].transform(features)
            
            # Make prediction
            prediction = models['model'].predict(features_scaled)[0]
            probabilities = models['model'].predict_proba(features_scaled)[0]
            
            confidence = float(np.max(probabilities))
            
            return {
                'prediction': int(prediction),
                'confidence': confidence,
                'risk_level': 'High' if prediction == 1 else 'Low',
                'probability_heart_disease': float(probabilities[1]) if len(probabilities) > 1 else confidence
            }
        
        except Exception as e:
            print(f"❌ Heart disease prediction error: {e}")
            return None
    
    def get_hospital_recommendations(self, location, condition_type):
        """Get manual Google Maps links for hospitals"""
        print(f"🗺️ Creating manual Google Maps search for {location}...")
        return self.hospital_finder.find_hospitals_manual(location, condition_type)

# Load and initialize everything
def load_specialist_models():
    """Load the diabetes and heart disease specialist models"""
    try:
        # Load diabetes model components
        diabetes_model = joblib.load('knn_diabetes_model.joblib')
        gender_encoder = joblib.load('gender_encoder.joblib')
        smoking_encoder = joblib.load('smoking_encoder.joblib')
        
        # Load heart disease model components 
        heart_model = joblib.load('heart_disease_xgb_model.joblib')
        heart_scaler = joblib.load('heart_xgb_scaler.joblib')
        
        print("✅ All specialist models loaded successfully!")
        
        return {
            'diabetes': {
                'model': diabetes_model,
                'gender_encoder': gender_encoder,
                'smoking_encoder': smoking_encoder
            },
            'heart': {
                'model': heart_model,
                'scaler': heart_scaler
            }
        }
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return None

def load_trained_triage_agent():
    """Load the medical triage agent"""
    try:
        saved_model = joblib.load('mednermodel.joblib')
        print("✅ Loading saved MedicalTriageAgent model...")
        
        agent = LoadedMedicalTriageAgent(saved_model)
        print("✅ MedicalTriageAgent loaded successfully from saved model")
        return agent
        
    except FileNotFoundError:
        print("⚠️ Saved model not found. Please ensure the model was saved properly.")
        return None
    except Exception as e:
        print(f"❌ Error loading triage agent: {e}")
        return None

# Initialize everything and create the enhanced_agent
def initialize_medical_agents():
    """Initialize all AI agents"""
    try:
        # Load models
        specialist_models = load_specialist_models()
        triage_agent = load_trained_triage_agent()
        
        if triage_agent and specialist_models:
            enhanced_agent = EnhancedMedicalSpecialistAgent(
                triage_agent=triage_agent,
                specialist_models=specialist_models
            )
            print("✅ Enhanced Medical Specialist Agent ready!")
            return enhanced_agent
        else:
            print("❌ Cannot initialize enhanced agent. Please check triage agent and specialist models.")
            return None
    except Exception as e:
        print(f"❌ Error initializing agents: {e}")
        return None

# Create the global enhanced_agent instance
enhanced_agent = initialize_medical_agents()
