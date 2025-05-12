import os
import subprocess
import google.auth.transport.requests


# # Add Google Cloud SDK to PATH
# os.environ['PATH'] = r'C:\Users\Qobi\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin;' + os.environ['PATH']

# # Function to run gcloud command and get access token local
# def get_access_token():
#     result = subprocess.run(['gcloud', 'auth', 'print-access-token'], capture_output=True, text=True, shell=True)
#     return result.stdout.strip()

# Function to get access token using service account credentials on GCP
def get_access_token():
    credentials, project = google.auth.default()
    auth_req = google.auth.transport.requests.Request()
    credentials.refresh(auth_req)
    return credentials.token