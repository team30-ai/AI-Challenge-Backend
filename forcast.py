from flask import request, jsonify, Blueprint
import pandas as pd
from prophet import Prophet
from datetime import datetime, timedelta
import os
import json
import tempfile
import shutil
from google.cloud import storage
import pickle

forcast_bp = Blueprint('forecast', __name__)


BUCKET_NAME = "hospital-scraper-bucket"
DATA_FILE = "combined_hospital_data.csv"
MODELS_FOLDER = "models"


def download_from_gcp(bucket_name, source_file, local_folder):
    """Download single file from GCP bucket to local folder"""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_file)
    
    if not os.path.exists(local_folder):
        os.makedirs(local_folder)
    
    destination_file = os.path.join(local_folder, os.path.basename(source_file))
    blob.download_to_filename(destination_file)
    
    return destination_file

def upload_to_gcp(bucket_name, source_file, destination_blob_name):
    """Upload a file to GCP bucket"""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file)

def check_model_exists(hospital_name):
    """Check if a trained model exists for this hospital"""
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    model_path = f"{MODELS_FOLDER}/{hospital_name.lower().replace(' ', '_')}_model.pkl"
    return storage.Blob(bucket=bucket, name=model_path).exists(storage_client)


def load_csv_data(csv_file_path):
    """Load a single CSV file with hospital data"""
    df = pd.read_csv(csv_file_path)
    

    try:
        df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    except:
        try:
            df["date"] = pd.to_datetime(df["date"])
        except Exception as e:
            raise ValueError(f"Failed to parse date column: {str(e)}")
    
    return df

def clean_data(df):
    """Clean the hospital data"""
    df = df.copy()
    for col in ['ed_trolleys', 'ward_trolleys', 'total_trolleys']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df

def get_hospital_list(df):
    """Return list of all hospitals in the dataset"""
    return df['hospital'].unique().tolist()


def train_and_save_model(df, hospital_name):
    """Train and save Prophet model for a hospital"""
    df_hosp = df[df["hospital"].str.lower() == hospital_name.lower()]
    if df_hosp.empty:
        raise ValueError(f"No data found for hospital: {hospital_name}")
    
    df_hosp = df_hosp[["date", "total_trolleys"]].rename(columns={"date": "ds", "total_trolleys": "y"})
    

    model = Prophet(daily_seasonality=True)
    model.fit(df_hosp)
   
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp:
        model_path = temp.name
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    

    destination_blob = f"{MODELS_FOLDER}/{hospital_name.lower().replace(' ', '_')}_model.pkl"
    upload_to_gcp(BUCKET_NAME, model_path, destination_blob)
    

    os.unlink(model_path)
    
    return model

def load_model(hospital_name):
    """Load a saved Prophet model from GCP"""

    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    model_blob = f"{MODELS_FOLDER}/{hospital_name.lower().replace(' ', '_')}_model.pkl"
    

    temp_dir = tempfile.mkdtemp()
    local_model_path = os.path.join(temp_dir, f"{hospital_name.lower().replace(' ', '_')}_model.pkl")
 
    bucket.blob(model_blob).download_to_filename(local_model_path)
    

    with open(local_model_path, 'rb') as f:
        model = pickle.load(f)
    

    shutil.rmtree(temp_dir)
    
    return model

def forecast_hospital(df, hospital_name, days=6):
    """Forecast hospital trolley numbers for specified days"""

    if check_model_exists(hospital_name):
        model = load_model(hospital_name)
    else:
        model = train_and_save_model(df, hospital_name)
    

    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    

    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    forecast = forecast[forecast["ds"] > today]
    forecast = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].head(days)
    
    return forecast

def get_forecast_json(forecast_df, hospital_name):
    """Convert forecast dataframe to JSON format"""
    forecast_json = []
    for _, row in forecast_df.iterrows():
        forecast_json.append({
            "date": row["ds"].strftime("%Y-%m-%d"),
            "yhat": round(max(0, row["yhat"]), 2),  
            "yhat_lower": round(max(0, row["yhat_lower"]), 2),
            "yhat_upper": round(max(0, row["yhat_upper"]), 2),
            "hospital": hospital_name
        })
    return forecast_json


@forcast_bp.route('/<hospital_name>', methods=['GET'])
def get_hospital_forecast(hospital_name):
    """Get forecast for a specific hospital"""
    try:
      
        temp_dir = tempfile.mkdtemp()
        
     
        csv_file_path = download_from_gcp(BUCKET_NAME, DATA_FILE, temp_dir)
        

        df = load_csv_data(csv_file_path)
        df = clean_data(df)

        forecast_df = forecast_hospital(df, hospital_name)
        forecast_json = get_forecast_json(forecast_df, hospital_name)
        

        shutil.rmtree(temp_dir)
        
        return jsonify(forecast_json)
    
    except Exception as e:

        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return jsonify({"error": str(e)}), 400

@forcast_bp.route('/all', methods=['GET'])
def get_all_hospitals_forecast():
    """Get forecast for all hospitals"""
    try:

        temp_dir = tempfile.mkdtemp()
        

        csv_file_path = download_from_gcp(BUCKET_NAME, DATA_FILE, temp_dir)
        

        df = load_csv_data(csv_file_path)
        df = clean_data(df)
        

        hospitals = get_hospital_list(df)

        all_forecasts = []
        for hospital in hospitals:
            try:
                forecast_df = forecast_hospital(df, hospital)
                hospital_forecast = get_forecast_json(forecast_df, hospital)
                all_forecasts.extend(hospital_forecast)
            except Exception as hospital_error:
                print(f"Error processing {hospital}: {str(hospital_error)}")
                continue
        

        shutil.rmtree(temp_dir)
        
        return jsonify(all_forecasts)
    
    except Exception as e:

        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return jsonify({"error": str(e)}), 400

@forcast_bp.route('/hospitals', methods=['GET'])
def get_hospitals():
    """Get list of all hospitals"""
    try:

        temp_dir = tempfile.mkdtemp()
        

        csv_file_path = download_from_gcp(BUCKET_NAME, DATA_FILE, temp_dir)
        

        df = load_csv_data(csv_file_path)
        

        hospitals = get_hospital_list(df)
        

        shutil.rmtree(temp_dir)
        
        return jsonify({"hospitals": hospitals})
    
    except Exception as e:

        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return jsonify({"error": str(e)}), 400