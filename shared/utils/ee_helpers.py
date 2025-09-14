# Earth Engine init and download helpers
import ee, os
import logging
from retry import retry
from concurrent.futures import ThreadPoolExecutor

# Configuration constants
TARGET_SCALE_METERS = 9265
DEFAULT_CRS = "EPSG:3857"
EE_SERVICE_ACCOUNT = "signature-work@signature-work-403906.iam.gserviceaccount.com"

def init_ee():
    service_account = "signature-work@signature-work-403906.iam.gserviceaccount.com"
    
    # Find the correct path to the JSON key file
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_key_path = os.path.join(current_dir, "../../data-pipeline/config/gee_key.json")
    
    if not os.path.exists(json_key_path):
        # Try alternative paths
        alt_paths = [
            "../config/gee_key.json",
            "../../config/gee_key.json", 
            "../data-pipeline/config/gee_key.json"
        ]
        for alt_path in alt_paths:
            test_path = os.path.join(current_dir, alt_path)
            if os.path.exists(test_path):
                json_key_path = test_path
                break
    
    print(f"Using GEE key file: {json_key_path}")
    ee.Initialize(
        ee.ServiceAccountCredentials(service_account, json_key_path),
        opt_url="https://earthengine-highvolume.googleapis.com"
    )

@retry(tries=5, delay=2, backoff=2)
def get_download_url(img, region, desc):
    return img.getDownloadUrl({
        "description": desc,
        "region": region,
        "crs": DEFAULT_CRS,
        "scale": TARGET_SCALE_METERS,
        "fileFormat": "GEO_TIFF"
    })

session = None

@retry(tries=3, delay=5, backoff=2)
def fetch_url(url_and_output):
    """
    Download a file from a URL (for direct downloads like SoilGrids).
    
    Args:
        url_and_output: Tuple of (url, output_path)
    """
    global session
    import requests
    if session is None:
        session = requests.Session()
    
    url, outp = url_and_output
    if os.path.exists(outp):
        return
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    
    try:
        r = session.get(url, stream=True, timeout=(10, 900))
        r.raise_for_status()
        with open(outp, 'wb') as f:
            for chunk in r.iter_content(1024 * 1024):
                f.write(chunk)
    except requests.exceptions.HTTPError as e:
        if "503" in str(e):
            print(f"⚠️  Google Earth Engine service temporarily unavailable (503). Retrying...")
            raise  # Let retry decorator handle it
        else:
            print(f"❌ HTTP Error downloading {outp}: {e}")
            raise
    except Exception as e:
        print(f"❌ Error downloading {outp}: {e}")
        raise

@retry(tries=3, delay=5, backoff=2)
def fetch_task(task, output_dir):
    """
    Wait for an Earth Engine task to complete and download the result.
    
    Args:
        task: Earth Engine Task object
        output_dir: Directory to save the downloaded file
    """
    import requests
    import time
    import zipfile
    global session
    
    if session is None:
        session = requests.Session()
    
    # Wait for task to complete
    print(f"Waiting for task {task.id} to complete...")
    while task.active():
        time.sleep(5)  # Check every 5 seconds instead of 10
        task_status = task.status()
        state = task_status.get('state', 'UNKNOWN')
        print(f"Task state: {state}")
        
        if state == 'FAILED':
            error_message = task_status.get('error_message', 'Unknown error')
            raise Exception(f"Task failed: {error_message}")
    
    # Get final status
    final_status = task.status()
    if final_status.get('state') != 'COMPLETED':
        raise Exception(f"Task ended with state: {final_status.get('state')}")
    
    # Get download URLs
    try:
        download_urls = final_status.get('destination_uris', [])
        if not download_urls:
            raise Exception("No download URLs found in completed task")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Download each file
        for url in download_urls:
            filename = url.split('/')[-1]
            output_path = os.path.join(output_dir, filename)
            
            if os.path.exists(output_path):
                print(f"File already exists: {output_path}")
                continue
                
            print(f"Downloading: {filename}")
            r = session.get(url, stream=True, timeout=(10, 900))
            r.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in r.iter_content(1024 * 1024):
                    f.write(chunk)
            
            print(f"Downloaded: {output_path}")
            
    except Exception as e:
        print(f"❌ Error downloading task results: {e}")
        raise