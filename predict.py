import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import holidays
import os
import tempfile
import traceback
from flask import Blueprint, request, jsonify, Flask
from google.cloud import storage

app = Flask(__name__)

predict_bp = Blueprint('predict', __name__)

BUCKET_NAME = "hospital-scraper-bucket"

def download_from_gcp_bucket(source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
        print(f"Downloaded {source_blob_name} to {destination_file_name}")
    except Exception as e:
        print(f"Error downloading from GCP bucket: {str(e)}")
        traceback.print_exc()
        raise

# 1. LOAD AND CLEAN HISTORICAL DATA
def load_clean_data():
    """Load data from GCP bucket and clean it"""
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    temp_file.close()
    
    try:
        # Download the CSV file from GCP
        download_from_gcp_bucket("combined_hospital_data.csv", temp_file.name)
        
        # Read and process the data
        df = pd.read_csv(temp_file.name, parse_dates=['date'])
        
        # Modify data cleaning to handle potential issues
        df = df.copy()  # Create a copy to avoid SettingWithCopyWarning
        
        # Fill NaN values more carefully
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)
        
        # Ensure date is datetime
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        
        # Sort values
        df = df.sort_values(by=['hospital', 'date'])
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        traceback.print_exc()
        raise
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

# 2. FEATURE ENGINEERING
def add_features(df):
    """Enhanced feature engineering with more robust handling"""
    df = df.copy()  # Create a copy to avoid SettingWithCopyWarning
    
    # Ensure date is datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])  # remove rows with invalid date parsing
    
    # Feature extraction
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Add Irish public holidays
    irish_holidays = holidays.Ireland(years=df['date'].dt.year.unique())
    df['is_holiday'] = df['date'].apply(lambda x: int(x in irish_holidays)).astype(int)
    
    # Handle prev_day_trolleys with more robust groupby
    df['prev_day_trolleys'] = df.groupby("hospital")['total_trolleys'].shift(1).fillna(df['total_trolleys'].mean())
    
    return df

# 3. LABEL CREATION
def add_label(df, threshold=30):
    """Create binary label for hospital crowding"""
    df = df.copy()  # Create a copy to avoid SettingWithCopyWarning
    df['is_crowded'] = (df['total_trolleys'] > threshold).astype(int)
    return df

# 4. TRAIN CLASSIFICATION MODEL
def train_model(df):
    """Train Random Forest Classifier"""
    features = ['day_of_week', 'month', 'is_weekend', 'is_holiday', 'prev_day_trolleys']
    
    # Ensure features are numeric and handle potential NaNs
    X = df[features].fillna(0)
    y = df['is_crowded']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Print model performance
    y_pred = clf.predict(X_test)
    print("Model Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return clf

# 5. SAVE MODEL
def save_model(model, blob_name="models/crowding_model.pkl"):
    """Save model to GCP bucket"""
    # First save to a temporary local file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl')
    temp_file.close()
    
    try:
        # Save model to the temp file
        joblib.dump(model, temp_file.name)
        
        # Upload to GCP bucket
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(temp_file.name)
        print(f"Model saved to gs://{BUCKET_NAME}/{blob_name}")
        return f"gs://{BUCKET_NAME}/{blob_name}"
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        traceback.print_exc()
        return None
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

# 6. LOAD MODEL
def load_model(blob_name="models/crowding_model.pkl"):
    """Load model from GCP bucket"""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl')
    temp_file.close()
    
    try:
        # Download from GCP bucket
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(blob_name)
        
        # Check if model exists in bucket
        if not blob.exists():
            print(f"No model found at gs://{BUCKET_NAME}/{blob_name}")
            return None
        
        # Download model to temp file
        blob.download_to_filename(temp_file.name)
        print(f"Model downloaded from gs://{BUCKET_NAME}/{blob_name}")
        
        # Load the model
        model = joblib.load(temp_file.name)
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        traceback.print_exc()
        return None
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

# 7. PREDICT FUNCTION
def predict_crowding(hospital, target_date, full_df, model):
    """Predict hospital crowding for a specific hospital and date"""
    # Get the most recent data for the specified hospital
    hospital_data = full_df[full_df['hospital'] == hospital]
    
    if hospital_data.empty:
        raise ValueError(f"No data found for hospital: {hospital}")
    
    last_known = hospital_data.sort_values(by='date').iloc[-1]
    prev_trolley = last_known['total_trolleys']
    
    date_obj = pd.to_datetime(target_date)
    features = {
        'day_of_week': date_obj.dayofweek,
        'month': date_obj.month,
        'is_weekend': int(date_obj.dayofweek in [5, 6]),
        'is_holiday': int(date_obj in holidays.Ireland()),
        'prev_day_trolleys': prev_trolley
    }
    
    input_df = pd.DataFrame([features])
    pred_prob = model.predict_proba(input_df)[0]
    will_be_crowded = bool(model.predict(input_df)[0])
    
    return {
        "date": target_date,
        "hospital": hospital,
        "will_be_crowded": will_be_crowded,
        "confidence": round(float(pred_prob[1]), 2)
    }

# Initialize model and data (called once during app startup)
def initialize():
    """Initialize the model and data"""
    print("Initializing model and data...")
    
    # Always load the latest data
    print("Loading data from GCP bucket...")
    try:
        df = load_clean_data()
    except Exception as e:
        print(f"Error loading data from GCP bucket: {str(e)}")
        # For local development, fallback to local CSV
        if os.path.exists("combined_hospital_data.csv"):
            print("Falling back to local CSV file")
            df = pd.read_csv("combined_hospital_data.csv", parse_dates=['date'])
            
            # Same cleaning steps as in load_clean_data()
            df = df.copy()
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].fillna(0)
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])
            df = df.sort_values(by=['hospital', 'date'])
        else:
            raise ValueError("No hospital data available")
    
    # Feature engineering and label creation
    df = add_features(df)
    df = add_label(df, threshold=30)
    
    # Try to load the model from GCP bucket
    model = load_model("models/crowding_model.pkl")
    
    # If model doesn't exist, train a new one and save it
    if model is None:
        print("Training new model...")
        model = train_model(df)
        save_model(model, "models/crowding_model.pkl")
    
    return model, df

# API ENDPOINTS
@predict_bp.route('/tomorrow', methods=['POST'])
def predict():
    """Predict crowding for a single hospital tomorrow"""
    content = request.json
    hospital_name = content.get('hospital')
    
    # Validate hospital name
    if not hospital_name:
        return jsonify({"error": "Hospital name is required"}), 400
    
    # Calculate the target date (tomorrow)
    tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Get prediction
    try:
        prediction = predict_crowding(
            hospital=hospital_name,
            target_date=tomorrow,
            full_df=df,
            model=model
        )
        return jsonify(prediction)
    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@predict_bp.route('/hospitals', methods=['GET'])
def list_hospitals():
    """List all unique hospitals in the dataset"""
    hospitals = df['hospital'].unique().tolist()
    return jsonify({"hospitals": hospitals})

@predict_bp.route('/all', methods=['GET'])
def predict_all_hospitals():
    """Get crowding predictions for all hospitals for tomorrow"""
    # Calculate the target date (tomorrow)
    tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    
    try:
        # Get list of all hospitals
        hospitals = df['hospital'].unique().tolist()
        
        # Get predictions for each hospital
        predictions = []
        for hospital in hospitals:
            prediction = predict_crowding(
                hospital=hospital,
                target_date=tomorrow,
                full_df=df,
                model=model
            )
            predictions.append(prediction)
        
        # Sort by confidence of crowding (highest first)
        predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
        
        return jsonify({
            "date": tomorrow,
            "predictions": predictions
        })
    except Exception as e:
        print(f"Error making all hospital predictions: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# Register the blueprint
app.register_blueprint(predict_bp, url_prefix='/predict')

# On application start, initialize the model and data
print("Starting application...")
model, df = initialize()

