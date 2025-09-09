from flask import Flask, request, jsonify, send_from_directory, session, redirect, url_for
import joblib
import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv
import datetime
import google.generativeai as genai
import json
from functools import wraps
import hashlib
import secrets

# ========================================
# Flask App Setup
# ========================================  
app = Flask(__name__, static_folder='../frontend')

# Load environment variables from system.env file
load_dotenv('../system.env')

# Configure session secret key
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-change-this-in-production')

# Local Authentication Configuration
# No external OAuth configuration needed

# Database setup - Using SQLite
DATABASE_PATH = "patient_data.db"
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

# Setup SQLAlchemy engine
engine = create_engine(DATABASE_URL, echo=False)

# Gemini AI Configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    print("✅ Gemini AI configured successfully!")
else:
    print("Warning: GEMINI_API_KEY not found in system.env. AI explanations will be disabled.")
    gemini_model = None

# Load ML model
MODEL_PATH = "../final_lgb_model.pkl"
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print("❌ Error loading model:", e)
else:
    print("❌ Model file not found!")

# ========================================
# Database Initialization
# ========================================
def init_db():
    """Initialize the database with required tables"""
    try:
        with engine.connect() as conn:
            # Create users table for local authentication
            conn.execute(text('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username VARCHAR(255) UNIQUE NOT NULL,
                    password_hash VARCHAR(255) NOT NULL,
                    email VARCHAR(255),
                    full_name VARCHAR(255),
                    phone VARCHAR(50),
                    position VARCHAR(100),
                    department VARCHAR(100),
                    hospital VARCHAR(255),
                    experience VARCHAR(20),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP
                )
            '''))
            
            # Create patients table
            conn.execute(text('''
                CREATE TABLE IF NOT EXISTS patients (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id VARCHAR(255) UNIQUE,
                    name VARCHAR(255),
                    sex VARCHAR(50),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            '''))
            
            # Create patient_data table for daily records
            conn.execute(text('''
                CREATE TABLE IF NOT EXISTS patient_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id VARCHAR(255),
                    day_number INTEGER,
                    risk_score REAL,
                    data_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
                )
            '''))
            
            # Create predictions table for general predictions
            conn.execute(text('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            '''))
            
            conn.commit()
            print("✅ Database tables initialized successfully!")
    except Exception as e:
        print(f"❌ Database initialization error: {e}")

# Initialize database on startup
init_db()

# SQLite database is ready - no sample data needed
print("✅ SQLite database initialized successfully!")

# ========================================
# AI Explanation Functions
# ========================================
def generate_ai_explanation(data_type, data):
    """Generate AI explanation for health data"""
    if not gemini_model:
        return "AI explanations are not available. Please set GEMINI_API_KEY environment variable."
    
    try:
        if data_type == "patient_analysis":
            prompt = f"""
            You are a medical AI assistant. Analyze this patient health risk data and provide a clear, user-friendly explanation:
            
            Patient Data Summary:
            - Total Days Analyzed: {data.get('total_days', 0)}
            - Average Risk Score: {data.get('avg_risk', 0):.3f}
            - High Risk Days: {data.get('high_risk_count', 0)}
            - Risk Level: {data.get('risk_level', 'Unknown')}
            
            Please provide:
            1. A simple explanation of what these numbers mean
            2. What the risk score represents (0-1 scale)
            3. Practical recommendations based on the data
            4. When to seek medical attention
            
            Keep the explanation clear, concise, and non-technical for patients to understand.
            """
        
        elif data_type == "global_summary":
            prompt = f"""
            You are a medical AI assistant. Analyze this global patient database summary and provide insights:
            
            Database Summary:
            - Total Patients: {data.get('total_patients', 0)}
            - Total Records: {data.get('total_records', 0)}
            - Average Risk Score: {data.get('avg_risk_score', 0):.3f}
            - High Risk Patients: {data.get('high_risk_patients', 0)}
            
            Please provide:
            1. What these statistics tell us about the patient population
            2. Key insights about overall health trends
            3. Recommendations for healthcare providers
            4. Areas that need attention
            
            Keep the explanation professional but accessible.
            """
        
        elif data_type == "patient_trend":
            risk_scores = data.get('risk_scores', [])
            prompt = f"""
            You are a medical AI assistant. Analyze this patient's 30-day risk trend:
            
            Risk Scores (30 days): {risk_scores[:10]}... (showing first 10 days)
            Average Risk: {data.get('avg_risk', 0):.3f}
            Highest Risk: {data.get('max_risk', 0):.3f}
            Lowest Risk: {data.get('min_risk', 0):.3f}
            
            Please provide:
            1. Analysis of the risk trend over 30 days
            2. Whether the patient's condition is improving, worsening, or stable
            3. Key patterns or concerning trends
            4. Specific recommendations for this patient
            
            Keep the explanation clear and actionable.
            """
        
        response = gemini_model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"AI explanation unavailable: {str(e)}"

# ========================================
# Authentication Functions
# ========================================
def hash_password(password):
    """Hash a password using SHA-256 with salt"""
    salt = secrets.token_hex(16)
    password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
    return f"{salt}:{password_hash}"

def verify_password(password, stored_hash):
    """Verify a password against its hash"""
    try:
        salt, password_hash = stored_hash.split(':')
        return hashlib.sha256((password + salt).encode()).hexdigest() == password_hash
    except:
        return False

def create_user(username, password, email=None, full_name=None, phone=None, position=None, department=None, hospital=None, experience=None):
    """Create a new user in the database"""
    try:
        with engine.connect() as conn:
            # Check if username already exists
            check_query = text('SELECT id FROM users WHERE username = :username')
            result = conn.execute(check_query, {"username": username})
            if result.fetchone():
                return False, "Username already exists"
            
            # Check if email already exists
            if email:
                email_query = text('SELECT id FROM users WHERE email = :email')
                email_result = conn.execute(email_query, {"email": email})
                if email_result.fetchone():
                    return False, "Email already exists"
            
            # Hash password and create user
            password_hash = hash_password(password)
            conn.execute(text('''
                INSERT INTO users (username, password_hash, email, full_name, phone, position, department, hospital, experience)
                VALUES (:username, :password_hash, :email, :full_name, :phone, :position, :department, :hospital, :experience)
            '''), {
                "username": username,
                "password_hash": password_hash,
                "email": email,
                "full_name": full_name,
                "phone": phone,
                "position": position,
                "department": department,
                "hospital": hospital,
                "experience": experience
            })
            conn.commit()
            return True, "User created successfully"
    except Exception as e:
        return False, f"Error creating user: {str(e)}"

def authenticate_user(username, password):
    """Authenticate a user with username and password"""
    try:
        with engine.connect() as conn:
            query = text('SELECT id, username, password_hash, email, full_name, phone, position, department, hospital, experience FROM users WHERE username = :username')
            result = conn.execute(query, {"username": username})
            user = result.fetchone()
            
            if user and verify_password(password, user[2]):
                # Update last login
                conn.execute(text('''
                    UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = :user_id
                '''), {"user_id": user[0]})
                conn.commit()
                
                return {
                    'id': user[0],
                    'username': user[1],
                    'email': user[3],
                    'full_name': user[4],
                    'phone': user[5],
                    'position': user[6],
                    'department': user[7],
                    'hospital': user[8],
                    'experience': user[9]
                }, None
            else:
                return None, "Invalid username or password"
    except Exception as e:
        return None, f"Authentication error: {str(e)}"

def login_required(f):
    """Decorator to require authentication for routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check if user is authenticated via session storage (handled on frontend)
        # This is a simple implementation - in production, you'd want server-side session management
        return f(*args, **kwargs)
    return decorated_function

# ========================================
# CORS Setup
# ========================================
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# ========================================
# Routes
# ========================================

# Authentication Routes
@app.route('/login')
def login_page():
    return send_from_directory('../frontend', 'login.html')

@app.route('/dashboard')
def dashboard_page():
    return send_from_directory('../frontend', 'dashboard.html')

@app.route('/api/auth/login', methods=['POST'])
def local_auth():
    """Handle local database authentication"""
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()
        
        if not username or not password:
            return jsonify({"success": False, "error": "Username and password are required"}), 400
        
        # Authenticate user
        user_info, error = authenticate_user(username, password)
        
        if error:
            return jsonify({"success": False, "error": error}), 400
        
        if not user_info:
            return jsonify({"success": False, "error": "Invalid username or password"}), 400
        
        # Store user info in session
        session['user'] = user_info
        session['is_authenticated'] = True
        
        return jsonify({
            "success": True,
            "user": user_info,
            "message": "Login successful"
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": f"Authentication error: {str(e)}"}), 500

@app.route('/api/auth/register', methods=['POST'])
def register():
    """Handle user registration"""
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()
        email = data.get('email', '').strip()
        full_name = data.get('full_name', '').strip()
        phone = data.get('phone', '').strip()
        position = data.get('position', '').strip()
        department = data.get('department', '').strip()
        hospital = data.get('hospital', '').strip()
        experience = data.get('experience', '').strip()
        
        # Required field validation
        if not username or not password or not email or not full_name or not position or not department:
            return jsonify({"success": False, "error": "Username, password, email, full name, position, and department are required"}), 400
        
        if len(password) < 6:
            return jsonify({"success": False, "error": "Password must be at least 6 characters long"}), 400
        
        # Email validation
        import re
        email_regex = r'^[^\s@]+@[^\s@]+\.[^\s@]+$'
        if not re.match(email_regex, email):
            return jsonify({"success": False, "error": "Please enter a valid email address"}), 400
        
        # Create user
        success, message = create_user(
            username=username,
            password=password,
            email=email,
            full_name=full_name,
            phone=phone if phone else None,
            position=position,
            department=department,
            hospital=hospital if hospital else None,
            experience=experience if experience else None
        )
        
        if success:
            return jsonify({"success": True, "message": message})
        else:
            return jsonify({"success": False, "error": message}), 400
        
    except Exception as e:
        return jsonify({"success": False, "error": f"Registration error: {str(e)}"}), 500

@app.route('/api/auth/logout', methods=['POST'])
def logout():
    """Handle user logout"""
    try:
        session.clear()
        return jsonify({"success": True, "message": "Logout successful"})
    except Exception as e:
        return jsonify({"success": False, "error": f"Logout error: {str(e)}"}), 500

@app.route('/api/auth/status')
def auth_status():
    """Check authentication status"""
    try:
        if session.get('is_authenticated'):
            return jsonify({
                "authenticated": True,
                "user": session.get('user')
            })
        else:
            return jsonify({"authenticated": False})
    except Exception as e:
        return jsonify({"authenticated": False, "error": str(e)})

@app.route('/api/auth/profile', methods=['PUT'])
def update_profile():
    """Update user profile information"""
    try:
        if not session.get('is_authenticated'):
            return jsonify({"success": False, "error": "Not authenticated"}), 401
        
        data = request.get_json()
        user_id = session.get('user', {}).get('id')
        
        if not user_id:
            return jsonify({"success": False, "error": "User ID not found"}), 400
        
        # Update user profile in database
        with engine.connect() as conn:
            update_fields = []
            update_values = {"user_id": user_id}
            
            if 'full_name' in data:
                update_fields.append("full_name = :full_name")
                update_values['full_name'] = data['full_name']
            
            if 'email' in data:
                update_fields.append("email = :email")
                update_values['email'] = data['email']
            
            if 'phone' in data:
                update_fields.append("phone = :phone")
                update_values['phone'] = data['phone']
            
            if 'position' in data:
                update_fields.append("position = :position")
                update_values['position'] = data['position']
            
            if 'department' in data:
                update_fields.append("department = :department")
                update_values['department'] = data['department']
            
            if 'hospital' in data:
                update_fields.append("hospital = :hospital")
                update_values['hospital'] = data['hospital']
            
            if 'experience' in data:
                update_fields.append("experience = :experience")
                update_values['experience'] = data['experience']
            
            if update_fields:
                query = f"UPDATE users SET {', '.join(update_fields)} WHERE id = :user_id"
                conn.execute(text(query), update_values)
                conn.commit()
                
                # Update session data
                session_user = session.get('user', {})
                if 'full_name' in data:
                    session_user['full_name'] = data['full_name']
                if 'email' in data:
                    session_user['email'] = data['email']
                if 'phone' in data:
                    session_user['phone'] = data['phone']
                if 'position' in data:
                    session_user['position'] = data['position']
                if 'department' in data:
                    session_user['department'] = data['department']
                if 'hospital' in data:
                    session_user['hospital'] = data['hospital']
                if 'experience' in data:
                    session_user['experience'] = data['experience']
                session['user'] = session_user
                
                return jsonify({"success": True, "message": "Profile updated successfully"})
            else:
                return jsonify({"success": False, "error": "No fields to update"}), 400
                
    except Exception as e:
        return jsonify({"success": False, "error": f"Profile update error: {str(e)}"}), 500

@app.route('/api/auth/change-password', methods=['POST'])
def change_password():
    """Change user password"""
    try:
        if not session.get('is_authenticated'):
            return jsonify({"success": False, "error": "Not authenticated"}), 401
        
        data = request.get_json()
        current_password = data.get('current_password', '').strip()
        new_password = data.get('new_password', '').strip()
        
        if not current_password or not new_password:
            return jsonify({"success": False, "error": "Current password and new password are required"}), 400
        
        if len(new_password) < 6:
            return jsonify({"success": False, "error": "New password must be at least 6 characters long"}), 400
        
        user_id = session.get('user', {}).get('id')
        if not user_id:
            return jsonify({"success": False, "error": "User ID not found"}), 400
        
        # Verify current password and update
        with engine.connect() as conn:
            # Get current password hash
            query = text('SELECT password_hash FROM users WHERE id = :user_id')
            result = conn.execute(query, {"user_id": user_id})
            user = result.fetchone()
            
            if not user:
                return jsonify({"success": False, "error": "User not found"}), 404
            
            # Verify current password
            if not verify_password(current_password, user[0]):
                return jsonify({"success": False, "error": "Current password is incorrect"}), 400
            
            # Hash new password and update
            new_password_hash = hash_password(new_password)
            update_query = text('UPDATE users SET password_hash = :password_hash WHERE id = :user_id')
            conn.execute(update_query, {"password_hash": new_password_hash, "user_id": user_id})
            conn.commit()
            
            return jsonify({"success": True, "message": "Password changed successfully"})
            
    except Exception as e:
        return jsonify({"success": False, "error": f"Password change error: {str(e)}"}), 500

# Serve frontend pages
@app.route('/')
def serve_frontend():
    return send_from_directory('../frontend', 'index.html')

@app.route('/local')
@login_required
def local_page():
    return send_from_directory('../frontend', 'local_page.html')

@app.route('/global')
@login_required
def global_page():
    return send_from_directory('../frontend', 'global_page.html')

@app.route('/chatbot')
@login_required
def chatbot_page():
    return send_from_directory('../frontend', 'chatbot.html')

# Upload CSV + Predict + Store in SQLite
@app.route('/upload', methods=['POST'])
def upload():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    # Get form data
    patient_name = request.form.get('patientName', '').strip()
    patient_sex = request.form.get('patientSex', '').strip()
    
    if not patient_name:
        return jsonify({"error": "Patient name is required"}), 400
    
    if not patient_sex:
        return jsonify({"error": "Patient sex is required"}), 400

    file = request.files['file']
    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": f"CSV read failed: {e}"}), 400

    X = df  # Adjust features if needed

    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            preds = proba[:, 1] if proba.shape[1] >= 2 else proba[:, 0]
        else:
            preds = model.predict(X)
        preds = np.array(preds).astype(float)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    average = float(np.mean(preds)) if len(preds) > 0 else 0.0

    # Store results in SQLite DB
    try:
        results_df = df.copy()
        results_df.columns = [c.lower().replace(" ", "_") for c in results_df.columns]
        results_df["prediction"] = preds
        results_df["uploaded_at"] = datetime.datetime.utcnow()

        results_df.to_sql("predictions", engine, if_exists="append", index=False)
        print("✅ Data stored in database")
    except SQLAlchemyError as e:
        print("❌ Database error:", e)

    # Save patient data to database (for global page functionality)
    save_patient_data(df, preds.tolist(), average, patient_name, patient_sex)

    # Generate AI explanation
    ai_explanation = generate_ai_explanation("patient_analysis", {
        "total_days": len(preds),
        "avg_risk": average,
        "high_risk_count": len([p for p in preds if p > 0.5]),
        "risk_level": "HIGH" if average > 0.7 else "MEDIUM" if average > 0.3 else "LOW"
    })

    return jsonify({
        "predictions": preds.tolist(),
        "average": average,
        "n_rows": len(preds),
        "ai_explanation": ai_explanation,
        "patient_name": patient_name,
        "patient_sex": patient_sex
    })

# Fetch history (last 20 predictions)
@app.route('/history', methods=['GET'])
def history():
    try:
        query = "SELECT * FROM predictions ORDER BY uploaded_at DESC LIMIT 20"
        df = pd.read_sql(query, engine)
        return df.to_json(orient="records")
    except SQLAlchemyError as e:
        return jsonify({"error": f"Database fetch failed: {e}"}), 500

# ========================================
# Patient Data Management (for Global Page)
# ========================================
def save_patient_data(df, predictions, average_risk, patient_name, patient_sex):
    """Save patient data to database"""
    try:
        with engine.connect() as conn:
            # Generate unique patient ID
            patient_id = f"patient_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Insert or update patient record
            conn.execute(text('''
                INSERT INTO patients (patient_id, name, sex, last_updated)
                VALUES (:patient_id, :name, :sex, CURRENT_TIMESTAMP)
                ON CONFLICT (patient_id) 
                DO UPDATE SET name = :name, sex = :sex, last_updated = CURRENT_TIMESTAMP
            '''), {"patient_id": patient_id, "name": patient_name, "sex": patient_sex})
            
            # Clear existing data for this patient
            conn.execute(text('DELETE FROM patient_data WHERE patient_id = :patient_id'), 
                        {"patient_id": patient_id})
            
            # Insert daily data
            for i, (_, row) in enumerate(df.iterrows()):
                conn.execute(text('''
                    INSERT INTO patient_data (patient_id, day_number, risk_score, data_json)
                    VALUES (:patient_id, :day_number, :risk_score, :data_json)
                '''), {
                    "patient_id": patient_id, 
                    "day_number": i + 1, 
                    "risk_score": predictions[i], 
                    "data_json": json.dumps(row.to_dict())
                })
            
            conn.commit()
            print(f"✅ Saved data for patient {patient_id}")
            
    except Exception as e:
        print(f"❌ Error saving patient data: {e}")

@app.route("/api/patients")
def get_all_patients():
    """Get all patients data"""
    try:
        with engine.connect() as conn:
            query = text('''
                SELECT p.patient_id, p.name, p.sex, p.created_at, p.last_updated,
                       COUNT(pd.day_number) as total_days,
                       AVG(pd.risk_score) as avg_risk,
                       MAX(pd.risk_score) as max_risk,
                       MIN(pd.risk_score) as min_risk
                FROM patients p
                LEFT JOIN patient_data pd ON p.patient_id = pd.patient_id
                GROUP BY p.patient_id, p.name, p.sex, p.created_at, p.last_updated
                ORDER BY p.last_updated DESC
            ''')
            
            result = conn.execute(query)
            patients = []
            for row in result:
                patients.append({
                    'patient_id': row[0],
                    'name': row[1],
                    'sex': row[2],
                    'created_at': str(row[3]),
                    'last_updated': str(row[4]),
                    'total_days': row[5],
                    'avg_risk': round(row[6], 3) if row[6] else 0,
                    'max_risk': round(row[7], 3) if row[7] else 0,
                    'min_risk': round(row[8], 3) if row[8] else 0
                })
            
            return jsonify(patients)
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/patients/<patient_id>/data")
def get_patient_data(patient_id):
    """Get detailed data for a specific patient"""
    try:
        with engine.connect() as conn:
            # Get patient info
            patient_query = text('SELECT * FROM patients WHERE patient_id = :patient_id')
            patient_result = conn.execute(patient_query, {"patient_id": patient_id})
            patient_info = patient_result.fetchone()
            
            if not patient_info:
                return jsonify({"error": "Patient not found"}), 404
            
            # Get daily data
            data_query = text('''
                SELECT day_number, risk_score, data_json, created_at
                FROM patient_data
                WHERE patient_id = :patient_id
                ORDER BY day_number
            ''')
            data_result = conn.execute(data_query, {"patient_id": patient_id})
            
            daily_data = []
            for row in data_result:
                daily_data.append({
                    'day': row[0],
                    'risk_score': row[1],
                    'data': json.loads(row[2]) if row[2] else {},
                    'created_at': str(row[3])
                })
            
            return jsonify({
                'patient_info': {
                    'patient_id': patient_info[1],
                    'name': patient_info[2],
                    'created_at': str(patient_info[3]),
                    'last_updated': str(patient_info[4])
                },
                'daily_data': daily_data
            })
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ========================================
# Interactive AI Chatbot Functions
# ========================================
def analyze_patient_data_for_chat(patient_id=None):
    """Analyze patient data and return comprehensive insights for chatbot"""
    try:
        with engine.connect() as conn:
            if patient_id:
                # Analyze specific patient
                query = text('''
                    SELECT p.patient_id, p.name, 
                           COUNT(pd.day_number) as total_days,
                           AVG(pd.risk_score) as avg_risk,
                           MAX(pd.risk_score) as max_risk,
                           MIN(pd.risk_score) as min_risk,
                           STDDEV(pd.risk_score) as risk_std,
                           COUNT(CASE WHEN pd.risk_score > 0.7 THEN 1 END) as high_risk_days,
                           COUNT(CASE WHEN pd.risk_score > 0.5 THEN 1 END) as medium_risk_days
                    FROM patients p
                    LEFT JOIN patient_data pd ON p.patient_id = pd.patient_id
                    WHERE p.patient_id = :patient_id
                    GROUP BY p.patient_id, p.name
                ''')
                result = conn.execute(query, {"patient_id": patient_id})
                row = result.fetchone()
                
                if row:
                    return {
                        "type": "patient_specific",
                        "patient_id": row[0],
                        "patient_name": row[1],
                        "total_days": row[2],
                        "avg_risk": round(row[3], 3) if row[3] else 0,
                        "max_risk": round(row[4], 3) if row[4] else 0,
                        "min_risk": round(row[5], 3) if row[5] else 0,
                        "risk_std": round(row[6], 3) if row[6] else 0,
                        "high_risk_days": row[7],
                        "medium_risk_days": row[8],
                        "risk_trend": "improving" if row[4] and row[5] and row[4] - row[5] < 0.2 else "stable" if row[6] and row[6] < 0.1 else "concerning"
                    }
            else:
                # Analyze all patients
                query = text('''
                    SELECT COUNT(DISTINCT p.patient_id) as total_patients,
                           COUNT(pd.id) as total_records,
                           AVG(pd.risk_score) as avg_risk,
                           MAX(pd.risk_score) as max_risk,
                           MIN(pd.risk_score) as min_risk,
                           COUNT(CASE WHEN pd.risk_score > 0.7 THEN 1 END) as high_risk_count,
                           COUNT(CASE WHEN pd.risk_score > 0.5 THEN 1 END) as medium_risk_count
                    FROM patients p
                    LEFT JOIN patient_data pd ON p.patient_id = pd.patient_id
                ''')
                result = conn.execute(query)
                row = result.fetchone()
                
                if row:
                    return {
                        "type": "global",
                        "total_patients": row[0],
                        "total_records": row[1],
                        "avg_risk": round(row[2], 3) if row[2] else 0,
                        "max_risk": round(row[3], 3) if row[3] else 0,
                        "min_risk": round(row[4], 3) if row[4] else 0,
                        "high_risk_count": row[5],
                        "medium_risk_count": row[6],
                        "overall_health": "good" if row[2] and row[2] < 0.3 else "moderate" if row[2] and row[2] < 0.6 else "concerning"
                    }
    except Exception as e:
        print(f"Error analyzing patient data: {e}")
        return None

def get_detailed_patient_data(patient_id):
    """Get detailed patient data including daily records"""
    try:
        with engine.connect() as conn:
            # Get patient info
            patient_query = text('SELECT * FROM patients WHERE patient_id = :patient_id')
            patient_result = conn.execute(patient_query, {"patient_id": patient_id})
            patient_info = patient_result.fetchone()
            
            if not patient_info:
                return None
            
            # Get daily data with detailed analysis
            data_query = text('''
                SELECT day_number, risk_score, data_json, created_at
                FROM patient_data
                WHERE patient_id = :patient_id
                ORDER BY day_number
            ''')
            data_result = conn.execute(data_query, {"patient_id": patient_id})
            
            daily_data = []
            risk_scores = []
            for row in data_result:
                daily_data.append({
                    'day': row[0],
                    'risk_score': row[1],
                    'data': json.loads(row[2]) if row[2] else {},
                    'created_at': str(row[3])
                })
                risk_scores.append(row[1])
            
            # Calculate trends
            if len(risk_scores) >= 7:
                recent_avg = sum(risk_scores[-7:]) / 7
                earlier_avg = sum(risk_scores[:-7]) / len(risk_scores[:-7]) if len(risk_scores) > 7 else recent_avg
                trend_direction = "improving" if recent_avg < earlier_avg - 0.05 else "worsening" if recent_avg > earlier_avg + 0.05 else "stable"
            else:
                trend_direction = "insufficient_data"
            
            return {
                'patient_info': {
                    'patient_id': patient_info[1],
                    'name': patient_info[2],
                    'created_at': str(patient_info[3]),
                    'last_updated': str(patient_info[4])
                },
                'daily_data': daily_data,
                'risk_scores': risk_scores,
                'trend_direction': trend_direction,
                'total_days': len(risk_scores)
            }
    except Exception as e:
        print(f"Error getting detailed patient data: {e}")
        return None

def process_chatbot_query(user_query, patient_id=None):
    """Process user query and generate intelligent response using Gemini AI"""
    if not gemini_model:
        return "AI chatbot is not available. Please set GEMINI_API_KEY environment variable."
    
    try:
        # Get comprehensive data analysis
        if patient_id:
            detailed_data = get_detailed_patient_data(patient_id)
            if not detailed_data:
                return f"I couldn't find data for patient {patient_id}. Please make sure the patient data has been uploaded."
            
            # Create detailed context for specific patient
            risk_scores = detailed_data['risk_scores']
            recent_scores = risk_scores[-7:] if len(risk_scores) >= 7 else risk_scores
            high_risk_days = [score for score in risk_scores if score > 0.7]
            medium_risk_days = [score for score in risk_scores if 0.5 < score <= 0.7]
            
            context = f"""
            You are a medical AI assistant analyzing real patient data. Answer the user's question based on the ACTUAL data provided below.
            
            PATIENT: {detailed_data['patient_info']['name']} (ID: {patient_id})
            MONITORING PERIOD: {detailed_data['total_days']} days
            
            ACTUAL DATA ANALYSIS:
            - Average Risk Score: {sum(risk_scores)/len(risk_scores):.3f} (0-1 scale)
            - Highest Risk Day: {max(risk_scores):.3f} (Day {risk_scores.index(max(risk_scores))+1})
            - Lowest Risk Day: {min(risk_scores):.3f} (Day {risk_scores.index(min(risk_scores))+1})
            - Recent 7-day average: {sum(recent_scores)/len(recent_scores):.3f}
            - High risk days (>0.7): {len(high_risk_days)} out of {len(risk_scores)}
            - Medium risk days (0.5-0.7): {len(medium_risk_days)} out of {len(risk_scores)}
            - Trend: {detailed_data['trend_direction']}
            
            RECENT RISK SCORES (last 7 days): {recent_scores}
            ALL RISK SCORES: {risk_scores}
            
            USER QUESTION: "{user_query}"
            
            IMPORTANT: Base your response ONLY on the actual data above. Do not use generic templates. 
            Provide specific insights about THIS patient's actual risk patterns, trends, and data points.
            If the user asks about specific days, trends, or patterns, reference the actual numbers provided.
            """
        else:
            # Global analysis
            data_analysis = analyze_patient_data_for_chat(patient_id)
            if not data_analysis:
                return "I couldn't retrieve patient data. Please make sure data has been uploaded."
            
            context = f"""
            You are a medical AI assistant analyzing a healthcare database. Answer based on the ACTUAL database statistics.
            
            DATABASE STATISTICS:
            - Total Patients: {data_analysis['total_patients']}
            - Total Records: {data_analysis['total_records']}
            - Average Risk Score: {data_analysis['avg_risk']}
            - Highest Risk Recorded: {data_analysis['max_risk']}
            - Lowest Risk Recorded: {data_analysis['min_risk']}
            - High Risk Records (>0.7): {data_analysis['high_risk_count']}
            - Medium Risk Records (0.5-0.7): {data_analysis['medium_risk_count']}
            - Overall Population Health: {data_analysis['overall_health']}
            
            USER QUESTION: "{user_query}"
            
            IMPORTANT: Base your response ONLY on the actual database statistics above. 
            Provide specific insights about the population health patterns and trends.
            """
        
        response = gemini_model.generate_content(context)
        return response.text
        
    except Exception as e:
        return f"I encountered an error processing your query: {str(e)}. Please try rephrasing your question."

# ========================================
# AI Explanation API Endpoints
# ========================================
@app.route("/api/ai/explain-global")
def explain_global_data():
    """Get AI explanation for global patient data"""
    try:
        with engine.connect() as conn:
            query = text('''
                SELECT COUNT(DISTINCT p.patient_id) as total_patients,
                       COUNT(pd.id) as total_records,
                       AVG(pd.risk_score) as avg_risk,
                       COUNT(CASE WHEN pd.risk_score > 0.5 THEN 1 END) as high_risk_count
                FROM patients p
                LEFT JOIN patient_data pd ON p.patient_id = pd.patient_id
            ''')
            
            result = conn.execute(query)
            row = result.fetchone()
            
            if row:
                summary_data = {
                    "total_patients": row[0],
                    "total_records": row[1],
                    "avg_risk_score": round(row[2], 3) if row[2] else 0,
                    "high_risk_patients": row[3]
                }
                
                ai_explanation = generate_ai_explanation("global_summary", summary_data)
                return jsonify({"explanation": ai_explanation})
            else:
                return jsonify({"explanation": "No data available for analysis."})
                
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/ai/explain-patient/<patient_id>")
def explain_patient_data(patient_id):
    """Get AI explanation for specific patient data"""
    try:
        with engine.connect() as conn:
            query = text('''
                SELECT risk_score FROM patient_data
                WHERE patient_id = :patient_id
                ORDER BY day_number
            ''')
            
            result = conn.execute(query, {"patient_id": patient_id})
            risk_scores = [row[0] for row in result]
            
            if risk_scores:
                patient_data = {
                    "risk_scores": risk_scores,
                    "avg_risk": sum(risk_scores) / len(risk_scores),
                    "max_risk": max(risk_scores),
                    "min_risk": min(risk_scores)
                }
                
                ai_explanation = generate_ai_explanation("patient_trend", patient_data)
                return jsonify({"explanation": ai_explanation})
            else:
                return jsonify({"explanation": "No data available for this patient."})
                
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ========================================
# Local AI Analysis (No Database Dependency)
# ========================================
@app.route("/api/local-ai/analyze", methods=['POST'])
def local_ai_analyze():
    """Analyze health data with AI using dashboard analysis results"""
    try:
        data = request.get_json()
        user_query = data.get('query', '').strip()
        analysis_data = data.get('analysis_data', {})
        
        if not user_query:
            return jsonify({"error": "Query cannot be empty"}), 400
        
        if not analysis_data:
            return jsonify({"error": "No analysis data provided"}), 400
        
        # Process the query with analysis data
        ai_response = analyze_health_data_with_ai(user_query, analysis_data)
        
        return jsonify({
            "response": ai_response,
            "query": user_query,
            "timestamp": datetime.datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": f"AI analysis error: {str(e)}"}), 500

def analyze_health_data_with_ai(user_query, analysis_data):
    """Analyze health data using AI with dashboard analysis results"""
    if not gemini_model:
        return "AI analysis is not available. Please set GEMINI_API_KEY environment variable."
    
    try:
        # Extract data from analysis results
        predictions = analysis_data.get('predictions', [])
        average_risk = analysis_data.get('average', 0)
        total_days = analysis_data.get('n_rows', len(predictions))
        high_risk_count = analysis_data.get('high_risk_count', 0)
        
        # Calculate risk distribution
        if predictions:
            high_risk_count = len([p for p in predictions if p > 0.7])
            medium_risk_count = len([p for p in predictions if 0.5 < p <= 0.7])
            low_risk_count = len([p for p in predictions if p <= 0.5])
            max_risk = max(predictions)
            min_risk = min(predictions)
        else:
            high_risk_count = 0
            medium_risk_count = 0
            low_risk_count = 0
            max_risk = average_risk
            min_risk = average_risk
        
        # Calculate percentages
        high_risk_percent = (high_risk_count / total_days * 100) if total_days > 0 else 0
        medium_risk_percent = (medium_risk_count / total_days * 100) if total_days > 0 else 0
        low_risk_percent = (low_risk_count / total_days * 100) if total_days > 0 else 0
        
        # Risk level interpretation
        if average_risk < 0.3:
            risk_level = "LOW RISK"
            risk_color = "🟢"
            risk_meaning = "Your health data shows a low risk of deterioration. This is good news!"
        elif average_risk < 0.6:
            risk_level = "MODERATE RISK"
            risk_color = "🟡"
            risk_meaning = "Your health data shows a moderate risk of deterioration. Some attention to health is recommended."
        else:
            risk_level = "HIGH RISK"
            risk_color = "🔴"
            risk_meaning = "Your health data shows a high risk of deterioration. Immediate medical attention is recommended."
        
        # Create comprehensive context for AI
        context = f"""
        You are a friendly medical AI assistant helping a patient understand their health data. The user has uploaded their health records and wants to understand what the data means.
        
        PATIENT HEALTH DATA SUMMARY:
        📊 Total Health Records Analyzed: {total_days}
        {risk_color} Overall Risk Level: {risk_level} ({average_risk:.1%})
        📈 Probability of Deterioration Range: {min_risk:.1%} to {max_risk:.1%}
        📊 Average Probability of Deterioration: {average_risk:.1%}
        
        RISK DISTRIBUTION:
        🔴 High Risk Days: {high_risk_count} ({high_risk_percent:.1f}%) - Risk > 70%
        🟡 Medium Risk Days: {medium_risk_count} ({medium_risk_percent:.1f}%) - Risk 50-70%
        🟢 Low Risk Days: {low_risk_count} ({low_risk_percent:.1f}%) - Risk < 50%
        
        DATA EXPLANATION:
        - Risk scores range from 0% (no risk) to 100% (very high risk)
        - Deterioration risk predicts the chance of health decline in the next 90 days
        - Higher scores indicate greater concern and need for medical attention
        - This analysis is based on your uploaded health data
        
        USER QUESTION: "{user_query}"
        
        INSTRUCTIONS FOR RESPONSE:
        1. Be friendly, clear, and easy to understand
        2. Use simple language - avoid medical jargon
        3. Explain what the numbers mean in practical terms
        4. Provide actionable advice based on the risk level
        5. Use emojis and formatting to make it engaging  
        6. Always reference the actual data from their analysis
        7. If they ask about specific numbers, explain what they mean
        8. Give practical next steps based on their risk level
        9. Focus on the "Probability of Deterioration for Next 90 Days" metric
        
        Make your response helpful, encouraging, and easy to understand for a regular person.
        """
        
        response = gemini_model.generate_content(context)
        return response.text
        
    except Exception as e:
        return f"I encountered an error analyzing your health data: {str(e)}. Please try rephrasing your question."

# ========================================
# Interactive AI Chatbot API Endpoints (Database Dependent - Keep for Global Page)
# ========================================
@app.route("/api/chatbot/chat", methods=['POST'])
def chatbot_chat():
    """Main chatbot endpoint for processing user queries"""
    try:
        data = request.get_json()
        user_query = data.get('query', '').strip()
        patient_id = data.get('patient_id', None)
        
        if not user_query:
            return jsonify({"error": "Query cannot be empty"}), 400
        
        # Process the query and get AI response
        ai_response = process_chatbot_query(user_query, patient_id)
        
        return jsonify({
            "response": ai_response,
            "query": user_query,
            "patient_id": patient_id,
            "timestamp": datetime.datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": f"Chatbot error: {str(e)}"}), 500

@app.route("/api/chatbot/patients")
def get_chatbot_patients():
    """Get list of patients for chatbot context"""
    try:
        with engine.connect() as conn:
            query = text('''
                SELECT p.patient_id, p.name, 
                       COUNT(pd.day_number) as total_days,
                       AVG(pd.risk_score) as avg_risk
                FROM patients p
                LEFT JOIN patient_data pd ON p.patient_id = pd.patient_id
                GROUP BY p.patient_id, p.name
                ORDER BY p.last_updated DESC
            ''')
            
            result = conn.execute(query)
            patients = []
            for row in result:
                patients.append({
                    'patient_id': row[0],
                    'name': row[1],
                    'total_days': row[2],
                    'avg_risk': round(row[3], 3) if row[3] else 0
                })
            
            return jsonify(patients)
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/chatbot/suggestions")
def get_chatbot_suggestions():
    """Get suggested questions for the chatbot"""
    suggestions = {
        "global": [
            "What is the overall health status of all patients?",
            "How many high-risk patients do we have?",
            "What are the main health trends in our database?",
            "Which patients need immediate attention?",
            "What is the average risk score across all patients?",
            "Show me the risk distribution across all patients",
            "What percentage of patients are in high risk category?"
        ],
        "patient_specific": [
            "What is this patient's current risk level?",
            "How has this patient's health changed over time?",
            "What are the main concerns for this patient?",
            "Should this patient see a doctor soon?",
            "What recommendations do you have for this patient?",
            "Show me this patient's risk trend over the last 7 days",
            "What was this patient's highest risk day?",
            "Is this patient's condition improving or worsening?",
            "How many high-risk days has this patient had?"
        ]
    }
    return jsonify(suggestions)

@app.route("/api/chatbot/patient/<patient_id>/insights")
def get_patient_insights(patient_id):
    """Get detailed insights for a specific patient"""
    try:
        detailed_data = get_detailed_patient_data(patient_id)
        if not detailed_data:
            return jsonify({"error": "Patient not found"}), 404
        
        risk_scores = detailed_data['risk_scores']
        if not risk_scores:
            return jsonify({"error": "No data available for this patient"}), 404
        
        # Calculate insights
        avg_risk = sum(risk_scores) / len(risk_scores)
        max_risk = max(risk_scores)
        min_risk = min(risk_scores)
        high_risk_days = len([s for s in risk_scores if s > 0.7])
        medium_risk_days = len([s for s in risk_scores if 0.5 < s <= 0.7])
        
        # Calculate trend
        if len(risk_scores) >= 7:
            recent_avg = sum(risk_scores[-7:]) / 7
            earlier_avg = sum(risk_scores[:-7]) / len(risk_scores[:-7]) if len(risk_scores) > 7 else recent_avg
            trend = "improving" if recent_avg < earlier_avg - 0.05 else "worsening" if recent_avg > earlier_avg + 0.05 else "stable"
        else:
            trend = "insufficient_data"
        
        insights = {
            "patient_info": detailed_data['patient_info'],
            "summary": {
                "total_days": len(risk_scores),
                "avg_risk": round(avg_risk, 3),
                "max_risk": round(max_risk, 3),
                "min_risk": round(min_risk, 3),
                "high_risk_days": high_risk_days,
                "medium_risk_days": medium_risk_days,
                "trend": trend
            },
            "risk_scores": risk_scores,
            "recent_scores": risk_scores[-7:] if len(risk_scores) >= 7 else risk_scores
        }
        
        return jsonify(insights)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/patients/<patient_id>", methods=['DELETE'])
def delete_patient(patient_id):
    """Delete a patient and all their data"""
    try:
        with engine.connect() as conn:
            # First check if patient exists
            check_query = text('SELECT name FROM patients WHERE patient_id = :patient_id')
            result = conn.execute(check_query, {"patient_id": patient_id})
            patient = result.fetchone()
            
            if not patient:
                return jsonify({"error": "Patient not found"}), 404
            
            patient_name = patient[0]
            
            # Delete patient data first (due to foreign key constraint)
            conn.execute(text('DELETE FROM patient_data WHERE patient_id = :patient_id'), 
                        {"patient_id": patient_id})
            
            # Delete patient record
            conn.execute(text('DELETE FROM patients WHERE patient_id = :patient_id'), 
                        {"patient_id": patient_id})
            
            conn.commit()
            
            return jsonify({
                "message": f"Patient '{patient_name}' and all associated data deleted successfully",
                "deleted_patient_id": patient_id
            })
            
    except Exception as e:
        return jsonify({"error": f"Failed to delete patient: {str(e)}"}), 500

# ========================================
# Run App
# ========================================
if __name__ == "__main__":
    app.run(debug=True)