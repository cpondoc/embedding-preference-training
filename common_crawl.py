"""
Code to utilize Common Crawl.
"""
from collections import deque
import gzip
import json
import os
import requests
import shutil

FILE_NAME = "data/common-crawl/cc-index.paths.gz"
TEXT_FILE = "data/common-crawl/cc-index.txt"
INDICES_FOLDER = "data/common-crawl/cc-indices"
LINKS_FOLDER = "data/common-crawl/cc-links"

def unzip_index(zip_name, file_name):
    """
    Unzips the core file name.
    """
    with gzip.open(zip_name, 'rb') as f_in:
        with open(file_name, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
            
def obtain_individual_index(path):
    """
    Download specific .gz file for each index.
    """
    # Craft URL and make response
    url = "https://data.commoncrawl.org/" + path
    file_name = path[-12:]
    response = requests.get(url, stream=True)
    
    # Save if file is successful
    if response.status_code == 200:
        with open(INDICES_FOLDER + "/" + file_name, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f'File downloaded successfully as {file_name}')
    else:
        print(f'Failed to download file. Status code: {response.status_code}')
    
            
def read_index_file():
    """
    Read in all of the files, get all of the .gz files one by one
    """
    # Extract all lines from the .txt file
    with open(TEXT_FILE, 'r') as file:
        for line in file: 
            stripped_line = line.strip()
            
            # Process and save the .gz file
            if ".gz" in stripped_line:
                obtain_individual_index(stripped_line)

def unzip_indices():
    """
    Unzip all .gz index files.
    """
    # List all files
    try:
        files = os.listdir(INDICES_FOLDER)
        for file in files:
            
            # Unzip each
            zip_name = INDICES_FOLDER + "/" + file
            file_name = LINKS_FOLDER + "/" + file[:-3] + ".txt"
            unzip_index(zip_name, file_name)
    except:
        print("Unable to unzip indices.")
        
def process_links():
    """
    Process all of the link data, extract just the URL.
    """
    # Iterate through all link files
    files = os.listdir(LINKS_FOLDER)
    for file in files:
        
        # Open said file
        file_name = LINKS_FOLDER + "/" + file
        with open(file_name, 'r') as f:
            
            # For right now: just print first 100 lines
            for i, line in enumerate(f):                    
                    # Massage data
                    line_arr = deque(line.split(" "))
                    line_arr.popleft()
                    line_arr.popleft()
                    new_line = " ".join(line_arr)
                    
                    # Convert to JSON and print
                    data = json.loads(new_line)
                    print(data['url'])

if __name__ == "__main__": 
    unzip_indices()
    