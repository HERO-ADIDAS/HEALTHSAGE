# app.py - HealthSage Flask Backend

from flask import Flask, request, render_template, redirect, url_for, session
import pandas as pd
from datetime import datetime
import urllib.parse
import os

# Import your AI agents
try:
    from medical_agents import enhanced_agent
    print("✅ HealthSage AI System loaded successfully!")
except ImportError as e:
    print(f"❌ Error loading AI agents: {e}")
    enhanced_agent = None

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'  # Change this in production

@app.route('/')
def index():
    """Main page with symptom input"""
    # Clear any existing session data
    session.clear()
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_symptoms():
    """Analyze symptoms and show triage results"""
    symptoms = request.form.get('symptoms', '').strip()
    
    if not symptoms:
        return render_template('index.html', error="Please describe your symptoms to continue.")
    
    if not enhanced_agent:
        return render_template('error.html', 
                             error="HealthSage AI system unavailable. Please try again later.")
    
    try:
        # Use your actual triage agent
        result = enhanced_agent.triage_agent.classify(symptoms)
        
        # Store in session
        session['symptoms'] = symptoms
        session['triage_result'] = result
        
        return render_template('triage_results.html', 
                             symptoms=symptoms, 
                             result=result)
        
    except Exception as e:
        return render_template('error.html', 
                             error=f"Analysis failed: {str(e)}")

@app.route('/specialist/<condition_type>')
def specialist_form(condition_type):
    """Show specialist assessment form"""
    if 'triage_result' not in session:
        return redirect(url_for('index'))
    
    triage_result = session['triage_result']
    
    if condition_type == 'diabetes' and triage_result['predicted_disease'] == 'diabetes':
        return render_template('diabetes_form.html', triage_result=triage_result)
    elif condition_type == 'cardiac' and triage_result['predicted_disease'] == 'hypertension':
        return render_template('cardiac_form.html', triage_result=triage_result)
    else:
        return render_template('no_specialist.html', 
                             condition=triage_result['predicted_disease'],
                             triage_result=triage_result)

@app.route('/assess_diabetes', methods=['POST'])
def assess_diabetes():
    """Process diabetes assessment"""
    if 'triage_result' not in session:
        return redirect(url_for('index'))
    
    try:
        # Collect form data
        diabetes_data = {
            'gender': request.form.get('gender'),
            'age': int(request.form.get('age')),
            'smoking_history': request.form.get('smoking_history'),
            'bmi': float(request.form.get('bmi')),
            'HbA1c_level': float(request.form.get('hba1c')),
            'blood_glucose_level': int(request.form.get('glucose')),
            'hypertension': int(request.form.get('hypertension')),
            'heart_disease': int(request.form.get('heart_disease'))
        }
        
        # Use your actual diabetes prediction
        result = enhanced_agent.predict_diabetes_risk(diabetes_data)
        
        if result:
            session['specialist_result'] = result
            session['specialist_type'] = 'diabetes'
            return render_template('specialist_results.html', 
                                 result=result, 
                                 specialist_type='diabetes',
                                 triage_result=session['triage_result'])
        else:
            return render_template('error.html', 
                                 error="Diabetes assessment failed. Please try again.")
            
    except Exception as e:
        return render_template('error.html', 
                             error=f"Assessment error: {str(e)}")

@app.route('/assess_cardiac', methods=['POST'])
def assess_cardiac():
    """Process cardiac assessment"""
    if 'triage_result' not in session:
        return redirect(url_for('index'))
    
    try:
        # Collect form data
        cardiac_data = {
            'age': int(request.form.get('age')),
            'sex': int(request.form.get('sex')),
            'cp': int(request.form.get('cp')),
            'trestbps': int(request.form.get('trestbps')),
            'chol': int(request.form.get('chol')),
            'fbs': int(request.form.get('fbs')),
            'restecg': int(request.form.get('restecg')),
            'thalach': int(request.form.get('thalach')),
            'exang': int(request.form.get('exang')),
            'oldpeak': float(request.form.get('oldpeak')),
            'slope': int(request.form.get('slope')),
            'ca': int(request.form.get('ca')),
            'thal': int(request.form.get('thal'))
        }
        
        # Use your actual cardiac prediction
        result = enhanced_agent.predict_heart_disease_risk(cardiac_data)
        
        if result:
            session['specialist_result'] = result
            session['specialist_type'] = 'cardiac'
            return render_template('specialist_results.html', 
                                 result=result, 
                                 specialist_type='cardiac',
                                 triage_result=session['triage_result'])
        else:
            return render_template('error.html', 
                                 error="Cardiac assessment failed. Please try again.")
            
    except Exception as e:
        return render_template('error.html', 
                             error=f"Assessment error: {str(e)}")

@app.route('/report')
def report_form():
    """Show report generation form"""
    if 'triage_result' not in session:
        return redirect(url_for('index'))
    
    return render_template('report_form.html', 
                         triage_result=session['triage_result'],
                         specialist_result=session.get('specialist_result'),
                         specialist_type=session.get('specialist_type'))

@app.route('/generate_report', methods=['POST'])
def generate_report():
    """Generate final report with hospital recommendations"""
    if 'triage_result' not in session:
        return redirect(url_for('index'))
    
    location = request.form.get('location', '').strip()
    
    if not location:
        return render_template('report_form.html', 
                             error="Please enter your location for hospital recommendations.",
                             triage_result=session['triage_result'],
                             specialist_result=session.get('specialist_result'),
                             specialist_type=session.get('specialist_type'))
    
    try:
        # Get hospital recommendations
        predicted_disease = session['triage_result']['predicted_disease']
        hospitals = enhanced_agent.get_hospital_recommendations(location, predicted_disease)
        
        hospital_data = None
        if hospitals:
            hospital_data = hospitals[0]
        
        # Generate timestamp
        timestamp = datetime.now().strftime('%B %d, %Y at %I:%M %p')
        
        return render_template('final_report.html',
                             symptoms=session['symptoms'],
                             triage_result=session['triage_result'],
                             specialist_result=session.get('specialist_result'),
                             specialist_type=session.get('specialist_type'),
                             hospital_data=hospital_data,
                             location=location,
                             timestamp=timestamp)
        
    except Exception as e:
        return render_template('error.html', 
                             error=f"Report generation failed: {str(e)}")

@app.route('/download_report')
def download_report():
    """Download report as text file"""
    if 'triage_result' not in session:
        return redirect(url_for('index'))
    
    # Generate report content
    report_content = generate_report_text()
    
    # Return as downloadable file
    from flask import Response
    return Response(
        report_content,
        mimetype="text/plain",
        headers={"Content-disposition": f"attachment; filename=HealthSage_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"}
    )

def generate_report_text():
    """Generate text content for downloadable report"""
    content = []
    content.append("=" * 60)
    content.append("HEALTHSAGE - MEDICAL CONSULTATION REPORT")
    content.append("=" * 60)
    content.append(f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")
    content.append("")
    
    # Symptoms
    content.append("SYMPTOMS:")
    content.append(f"- {session['symptoms']}")
    content.append("")
    
    # Triage Results
    triage_result = session['triage_result']
    content.append("PRIMARY ASSESSMENT:")
    content.append(f"- Condition: {triage_result['predicted_disease'].replace('_', ' ').title()}")
    content.append(f"- Confidence: {triage_result['confidence']:.1%}")
    content.append(f"- Recommended Action: {triage_result['recommended_action']}")
    content.append("")
    
    # Specialist Results
    if 'specialist_result' in session:
        specialist_result = session['specialist_result']
        specialist_type = session['specialist_type']
        
        title = "DIABETES ASSESSMENT:" if specialist_type == 'diabetes' else "CARDIAC ASSESSMENT:"
        content.append(title)
        content.append(f"- Risk Level: {specialist_result['risk_level']}")
        content.append(f"- Confidence: {specialist_result['confidence']:.1%}")
        
        if specialist_type == 'diabetes':
            content.append(f"- Diabetes Probability: {specialist_result['probability_diabetes']:.1%}")
        else:
            content.append(f"- Heart Disease Probability: {specialist_result['probability_heart_disease']:.1%}")
        content.append("")
    
    # Disclaimer
    content.append("IMPORTANT DISCLAIMER:")
    content.append("- This is an AI-assisted preliminary assessment only")
    content.append("- NOT a substitute for professional medical diagnosis")
    content.append("- Seek immediate medical attention for emergency symptoms")
    content.append("- Always consult healthcare providers for final diagnosis")
    content.append("- Emergency: 108 (Ambulance) | 102 (Medical Emergency)")
    
    return "\n".join(content)

if __name__ == '__main__':
    print("🧠 Starting HealthSage Medical Consultation System...")
    app.run(debug=True, host='0.0.0.0', port=5000)
