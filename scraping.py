import requests
from flask import Blueprint
import pandas as pd
from datetime import datetime, timedelta
import os
from bs4 import BeautifulSoup
from google.cloud import storage
import argparse
from flask import Flask, request, jsonify


scraping_bp = Blueprint('scraping', __name__)

def direct_extract_hospitals(html_content=None, date=None):
    """
    A direct approach to extract data for all Irish hospitals by searching for specific text patterns.
    
    Args:
        html_content (str, optional): HTML content to parse. If None, fetches from website.
        date (str, optional): Date in format 'DD/MM/YYYY'. If None, uses today's date.
        
    Returns:
        dict: Dictionary containing data for all hospitals
    """
    # Format date if provided, otherwise use today's date
    if date is None:
        date = datetime.now().strftime('%d/%m/%Y')
    
    # If HTML content is not provided, fetch it from the website
    if html_content is None:
        # Format the date for the URL (e.g., 9%2F05%2F2025)
        date_parts = date.split('/')
        url_date = f"{date_parts[0]}%2F{date_parts[1]}%2F{date_parts[2]}"
        
        # Construct the URL
        url = f"https://uec.hse.ie/uec/TGAR.php?EDDATE={url_date}"
        
        # Send GET request
        response = requests.get(url)
        
        # Check if request was successful
        if response.status_code != 200:
            return {"error": f"Failed to fetch data: HTTP {response.status_code}"}
        
        html_content = response.text
    
    # Parse HTML content
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Convert the HTML to a string for easier pattern matching
    html_str = str(soup)
    
    # List of all hospitals to extract
    all_hospitals = [
        "Beaumont Hospital", "Cavan General Hospital", "Connolly Hospital", 
        "Louth County Hospital", "Mater Misericordiae University Hospital", 
        "National Orthopaedic Hospital Cappagh", "Our Lady of Lourdes Hospital", 
        "Our Lady's Hospital Navan", "CHI at Crumlin", "CHI at Tallaght", 
        "CHI at Temple Street", "MRH Mullingar", "MRH Portlaoise", "MRH Tullamore", 
        "Naas General Hospital", "St. James's Hospital", 
        "St. Luke's Radiation Oncology Network", "Tallaght University Hospital", 
        "National Rehabilitation Hospital", "St. Columcille's Hospital", 
        "St Luke's General Hospital Kilkenny", "St. Michael's Hospital", 
        "St. Vincent's University Hospital", "Tipperary University Hospital", 
        "UH Waterford", "Wexford General Hospital", "Bantry General Hospital", 
        "Cork University Hospital", "Mallow General Hospital", "Mercy University Hospital", 
        "South Infirmary Victoria University Hospital", "UH Kerry", "Ennis Hospital", 
        "Nenagh Hospital", "St. John's Hospital Limerick", "UH Limerick", 
        "Galway University Hospital", "Letterkenny University Hospital", 
        "Mayo University Hospital", "Portiuncula University Hospital", 
        "Roscommon University Hospital", "Sligo University Hospital"
    ]
    
    # Create dictionary to store results
    results = {hospital.lower().replace(" ", "_").replace("'", "").replace(".", ""): {} 
              for hospital in all_hospitals}
    
    # Function to extract data after finding a hospital section
    def extract_after_hospital(hospital_name, start_pos):
        # Get the portion of HTML after the hospital name
        relevant_html = html_str[start_pos:start_pos + 2000]  # Look at next 2000 chars
        
        # Parse this section
        section_soup = BeautifulSoup(relevant_html, 'html.parser')
        
        # Find all TD cells with specific classes or attributes
        td_cells = section_soup.find_all('td', class_=lambda c: c and (
            'hse-u-bg-color_hse-white-0' in c or 
            'hse-u-bg-color_hse-green-500' in c or 
            'hse-u-bg-color_hse-red-500' in c))
        
        # Extract values
        data = {
            "hospital": hospital_name,
            "date": date
        }
        
        # Check if we have enough cells
        if len(td_cells) >= 3:
            data["ed_trolleys"] = td_cells[0].text.strip()
            data["ward_trolleys"] = td_cells[1].text.strip()
            data["total_trolleys"] = td_cells[2].text.strip()
            
            if len(td_cells) >= 4:
                data["surge_open"] = td_cells[3].text.strip()
                
                # Look for delayed transfers in specific cells
                red_cells = section_soup.find_all('td', class_=lambda c: c and 'hse-u-bg-color_hse-red-500' in c)
                if red_cells:
                    data["delayed_transfers"] = red_cells[0].text.strip()
                else:
                    data["delayed_transfers"] = "N/A"
                
                # Extract the remaining cells if available
                if len(td_cells) >= 6:
                    data["over_24hrs_on_trolleys"] = td_cells[5].text.strip() if td_cells[5].text.strip() else "N/A"
                    
                    if len(td_cells) >= 7:
                        data["over_75yrs_over_24hrs"] = td_cells[6].text.strip() if td_cells[6].text.strip() else "N/A"
                    else:
                        data["over_75yrs_over_24hrs"] = "N/A"
                else:
                    data["over_24hrs_on_trolleys"] = "N/A"
                    data["over_75yrs_over_24hrs"] = "N/A"
            else:
                data["surge_open"] = "N/A"
                data["delayed_transfers"] = "N/A"
                data["over_24hrs_on_trolleys"] = "N/A"
                data["over_75yrs_over_24hrs"] = "N/A"
        else:
            data["ed_trolleys"] = "N/A"
            data["ward_trolleys"] = "N/A" 
            data["total_trolleys"] = "N/A"
            data["surge_open"] = "N/A"
            data["delayed_transfers"] = "N/A"
            data["over_24hrs_on_trolleys"] = "N/A"
            data["over_75yrs_over_24hrs"] = "N/A"
            
        return data
    
    # Iterate through all hospitals and extract data
    for hospital in all_hospitals:
        # Convert hospital name to dictionary key format
        key = hospital.lower().replace(" ", "_").replace("'", "").replace(".", "")
        
        # Find hospital position in HTML
        hospital_pos = html_str.find(hospital)
        
        if hospital_pos > -1:
            results[key] = extract_after_hospital(hospital, hospital_pos)
        else:
            results[key] = {"error": f"{hospital} data not found"}
    
    return results

def clean_na(data_dict):
    """Clean N/A values in the data dictionary"""
    for key in data_dict:
        if isinstance(data_dict[key], str) and data_dict[key].strip().upper() == "N/A":
            data_dict[key] = "0"
    return data_dict

def save_to_dataframe(data):
    """
    Convert the scraped data to a DataFrame.
    
    Args:
        data (dict): Dictionary containing the scraped data
        
    Returns:
        DataFrame: Pandas DataFrame with the hospital data
    """
    # Prepare data for DataFrame
    rows = []
    
    # Add data for all hospitals if available
    for hospital_key in data:
        if "error" not in data[hospital_key]:
            rows.append(data[hospital_key])
    
    if not rows:
        print("No valid data to save")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(rows)
    return df


def clean_hospital_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and standardizes hospital data:
    - Converts numeric columns to correct dtype
    - Fills missing values with 0
    - Ensures date is a proper datetime object
    """
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')

    # List of columns to convert to numeric
    numeric_columns = [
        'ed_trolleys',
        'ward_trolleys',
        'total_trolleys',
        'surge_open',
        'delayed_transfers',
        'over_24hrs_on_trolleys',
        'over_75yrs_over_24hrs'
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    # Optional: standardize column names (lowercase, underscores)
    df.columns = [col.lower().replace(" ", "_") for col in df.columns]

    return df

def upload_to_gcs(file_path, bucket_name, destination_blob_name):
    """
    Upload a file to Google Cloud Storage bucket
    
    Args:
        file_path (str): Path to the local file
        bucket_name (str): Name of the GCS bucket
        destination_blob_name (str): Name to assign to the GCS object
    """
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)

        blob.upload_from_filename(file_path)
        print(f"File {file_path} uploaded to {destination_blob_name} in bucket {bucket_name}")
        # Delete local file after successful upload
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Local file {file_path} deleted after upload.")
    except Exception as e:
        print(f"Error uploading to GCS: {e}")

def download_from_gcs(bucket_name, source_blob_name, destination_file_name):
    """
    Download a file from Google Cloud Storage bucket
    
    Args:
        bucket_name (str): Name of the GCS bucket
        source_blob_name (str): Name of the GCS object
        destination_file_name (str): Path to save the file locally
        
    Returns:
        bool: True if download was successful, False otherwise
    """
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        
        blob.download_to_filename(destination_file_name)
        print(f"Downloaded {source_blob_name} from bucket {bucket_name} to {destination_file_name}")
        return True
    except Exception as e:
        print(f"Error downloading from GCS (this may be normal for first run): {e}")
        return False

def process_hospital_data(mode='daily', start_date='01/02/2024', bucket_name=None, specific_date=None):
    """
    Process hospital data based on mode and parameters
    
    Args:
        mode (str): 'historical' or 'daily' or 'specific'
        start_date (str): Start date for historical mode (DD/MM/YYYY)
        bucket_name (str): GCS bucket name
        specific_date (str): Specific date to scrape (DD/MM/YYYY)
        
    Returns:
        dict: Results summary
    """
    if bucket_name is None:
        return {"error": "Bucket name is required"}
    
    # Calculate yesterday's date
    yesterday = datetime.now() - timedelta(days=1)
    yesterday_str = yesterday.strftime("%d/%m/%Y")
    
    # File names
    local_file = "combined_hospital_data.csv"
    gcs_blob_name = "combined_hospital_data.csv"
    
    # Try to download existing combined file from GCS
    file_exists = download_from_gcs(bucket_name, gcs_blob_name, local_file)
    
    # Initialize or load existing dataframe
    if file_exists:
        combined_df = pd.read_csv(local_file)
        print(f"Loaded existing data with {len(combined_df)} records")
    else:
        combined_df = pd.DataFrame()
        print("Creating new combined dataset")
    
    results = {"mode": mode, "bucket": bucket_name}
    
    if mode == 'historical':
        # For historical mode, scrape from start date to yesterday
        start_date_obj = datetime.strptime(start_date, "%d/%m/%Y")
        end_date = yesterday
        
        results["start_date"] = start_date
        results["end_date"] = yesterday_str
        results["dates_processed"] = []
        
        current_date = start_date_obj
        while current_date <= end_date:
            date_str = current_date.strftime("%d/%m/%Y")
            
            # Check if this date already exists in the dataframe
            if not combined_df.empty and date_str in combined_df['date'].values:
                print(f"Data for {date_str} already exists, skipping...")
                current_date += timedelta(days=1)
                continue
                
            print(f"\nScraping data for {date_str}...")
            hospital_data = direct_extract_hospitals(date=date_str)
            
            # Clean N/A values
            for hospital_key in hospital_data:
                if "error" not in hospital_data[hospital_key]:
                    hospital_data[hospital_key] = clean_na(hospital_data[hospital_key])
            
            # Convert to DataFrame and append
            daily_df = save_to_dataframe(hospital_data)
            if daily_df is not None:
                combined_df = pd.concat([combined_df, daily_df], ignore_index=True)
                print(f"Added {len(daily_df)} records for {date_str}")
                results["dates_processed"].append(date_str)
            
            # Move to next day
            current_date += timedelta(days=1)
    
    elif mode == 'specific':
        # For specific date mode
        if specific_date is None:
            return {"error": "Specific date is required for 'specific' mode"}
            
        date_str = specific_date
        results["date"] = date_str
        
        # Check if this date already exists in the dataframe
        if not combined_df.empty and date_str in combined_df['date'].values:
            print(f"Data for {date_str} already exists, skipping...")
            results["status"] = "skipped"
            return results
            
        print(f"\nScraping data for {date_str}...")
        hospital_data = direct_extract_hospitals(date=date_str)
        
        # Clean N/A values
        for hospital_key in hospital_data:
            if "error" not in hospital_data[hospital_key]:
                hospital_data[hospital_key] = clean_na(hospital_data[hospital_key])
        
        # Convert to DataFrame and append
        daily_df = save_to_dataframe(hospital_data)
        if daily_df is not None:
            combined_df = pd.concat([combined_df, daily_df], ignore_index=True)
            print(f"Added {len(daily_df)} records for {date_str}")
            results["status"] = "added"
            results["records_added"] = len(daily_df)
        else:
            print(f"No valid data found for {date_str}")
            results["status"] = "no_data"
    
    else:  # daily mode
        # For daily mode, only scrape yesterday's data
        print(f"\nScraping data for {yesterday_str}...")
        results["date"] = yesterday_str
        
        # Check if yesterday's data already exists
        if not combined_df.empty and yesterday_str in combined_df['date'].values:
            print(f"Data for {yesterday_str} already exists, nothing to do")
            results["status"] = "skipped"
        else:
            hospital_data = direct_extract_hospitals(date=yesterday_str)
            
            # Clean N/A values
            for hospital_key in hospital_data:
                if "error" not in hospital_data[hospital_key]:
                    hospital_data[hospital_key] = clean_na(hospital_data[hospital_key])
            
            # Convert to DataFrame and append
            daily_df = save_to_dataframe(hospital_data)
            if daily_df is not None:
                combined_df = pd.concat([combined_df, daily_df], ignore_index=True)
                print(f"Added {len(daily_df)} records for {yesterday_str}")
                results["status"] = "added"
                results["records_added"] = len(daily_df)
            else:
                print(f"No valid data found for {yesterday_str}")
                results["status"] = "no_data"
    
    # Save combined data locally
    if not combined_df.empty:
        
        combined_df = clean_hospital_dataframe(combined_df)
        combined_df.to_csv(local_file, index=False)
        print(f"Combined data saved to {local_file} with {len(combined_df)} total records")
        results["total_records"] = len(combined_df)
        
        # Upload to GCS
        upload_to_gcs(local_file, bucket_name, gcs_blob_name)
    else:
        print("No data to save")
        results["total_records"] = 0
    
    return results

# Flask endpoint
@scraping_bp.route('/data', methods=['POST'])
def scrape_endpoint():
    try:
        data = request.get_json() or {}
        mode = data.get('mode', 'daily')  # Default to daily mode
        start_date = data.get('start_date', '01/02/2024')
        bucket_name = data.get('bucket_name')
        specific_date = data.get('specific_date')
        
        if bucket_name is None:
            return jsonify({"error": "bucket_name is required"}), 400
            
        if mode not in ['daily', 'historical', 'specific']:
            return jsonify({"error": "Invalid mode. Choose 'daily', 'historical', or 'specific'"}), 400
            
        results = process_hospital_data(mode, start_date, bucket_name, specific_date)
        return jsonify(results)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Health check endpoint
@scraping_bp.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "service": "hospital-data-scraper"})

# Command line option still available
def main():
    parser = argparse.ArgumentParser(description='Hospital Data Scraper')
    parser.add_argument('--mode', choices=['historical', 'daily', 'specific'], default='daily',
                        help='Mode to run: historical (from start date to yesterday), daily (only yesterday), or specific (specific date)')
    parser.add_argument('--start_date', default='01/02/2024',
                        help='Start date for historical scraping (DD/MM/YYYY)')
    parser.add_argument('--bucket_name', required=True,
                        help='GCS bucket name for storing the data')
    parser.add_argument('--specific_date', 
                        help='Specific date to scrape (DD/MM/YYYY), only used with --mode specific')
    
    args = parser.parse_args()
    
    results = process_hospital_data(
        mode=args.mode, 
        start_date=args.start_date, 
        bucket_name=args.bucket_name,
        specific_date=args.specific_date
    )
    
    print(f"Results: {results}")
