import os
import requests

def download_model_from_github(model_url, model_path):
    try:
        # Request the model file from the GitHub release
        response = requests.get(model_url, stream=True)
        response.raise_for_status()  # Ensure we get a 200 OK response
        
        # Write the model to the file
        with open(model_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
        print(f"Model downloaded successfully and saved to {model_path}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the model: {e}")
        raise
