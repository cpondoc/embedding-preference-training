"""
Code to extract WARCs and use a random subset of them.
"""
import requests
import gzip
import shutil
import os

# Function to download a file
def download_file(url, output_path):
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            shutil.copyfileobj(response.raw, f)
        print(f"Downloaded: {output_path}")
    else:
        print(f"Failed to download {url}: {response.status_code}")
    return output_path

# Function to decompress a .gz file
def decompress_gz(file_path, output_path):
    print(f"Decompressing {file_path}...")
    with gzip.open(file_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"Decompressed to: {output_path}")
    return output_path


"""
Let's first get all the WARC paths.
"""
indices = []
with open('data/common-crawl/warc.paths.txt', 'r') as file:
    # Loop through each line in the file
    for line in file:
        # Process each line (you can modify this as needed)
        line = line.strip()  # .strip() removes leading/trailing whitespace
        if line[-3:] == ".gz":
            indices.append(line)

"""
Now let's process an individual WARC path.
"""
# Process path of WARC into usable things.
path = indices[0]
warc_paths_url = "https://data.commoncrawl.org/" + path
compressed_file = "data/common-crawl/" + path.split("/")[-1]

# Now download and compress
# download_file(warc_paths_url, compressed_file)

"""
Now that we have the decompressed file, let's try to read into it.
"""
from warcio.archiveiterator import ArchiveIterator
import os
from urllib.parse import urlparse
from io import BytesIO
import logging
from datetime import datetime
import re
from collections import defaultdict

def parse_http_header(content):
    """Parse HTTP headers from raw content."""
    try:
        # Split headers from body
        header_end = content.find(b'\r\n\r\n')
        if header_end == -1:
            return None, content
            
        headers = content[:header_end]
        body = content[header_end + 4:]
        
        # Parse headers
        headers_dict = {}
        for line in headers.decode('utf-8', 'ignore').split('\r\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                headers_dict[key.strip().lower()] = value.strip()
                
        return headers_dict, body
    except Exception as e:
        logging.error(f"Error parsing HTTP headers: {e}")
        return None, content

def clean_filename(url):
    """Create a safe filename from URL."""
    # Remove protocol and special characters
    name = re.sub(r'[^\w\-_.]', '_', url)
    # Limit length
    return name[:150] + '.html'

def extract_html_pages(warc_path, output_dir='extracted_html'):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Statistics
    stats = {
        'total_records': 0,
        'response_records': 0,
        'html_pages': 0,
        'other_content': 0
    }
    
    try:
        with open(warc_path, 'rb') as stream:
            for record in ArchiveIterator(stream):
                stats['total_records'] += 1
                
                if stats['total_records'] % 100 == 0:
                    print(f"Processed {stats['total_records']} records...")
                
                # Check if it's a response record
                if record.rec_headers.get_header('WARC-Type') == 'response':
                    stats['response_records'] += 1
                    target_uri = record.rec_headers.get_header('WARC-Target-URI')
                    
                    try:
                        # Get the raw content
                        content = record.content_stream().read()
                        print(content)
                        exit()
                        # Parse HTTP headers and body
                        headers, body = parse_http_header(content)
                        
                        if headers:
                            content_type = headers.get('content-type', '').lower()
                            # Check if it's HTML content
                            if 'text/html' in content_type:
                                try:
                                    # Decode the HTML content
                                    html_content = body.decode('utf-8', 'ignore')
                                    
                                    if html_content.strip():
                                        stats['html_pages'] += 1
                                        
                                        # Create filename from URL
                                        if target_uri:
                                            filename = clean_filename(target_uri)
                                            filepath = os.path.join(output_dir, filename)
                                            
                                            # Save the HTML content
                                            with open(filepath, 'w', encoding='utf-8') as f:
                                                f.write(html_content)
                                            
                                            print(f"\nSaved HTML from {target_uri}")
                                            print(f"Saved to: {filepath}")
                                            print("Content preview:")
                                            print(html_content[:500])
                                            print("-" * 80)
                                            
                                except Exception as e:
                                    logging.error(f"Error processing HTML for {target_uri}: {e}")
                            else:
                                stats['other_content'] += 1
                                print(f"\nSkipping non-HTML content type: {content_type}")
                                print(f"URL: {target_uri}")
                    
                    except Exception as e:
                        logging.error(f"Error processing record: {e}")
                        continue
    
    except Exception as e:
        logging.error(f"Error processing WARC file: {e}")
    
    # Print final statistics
    print("\nProcessing Complete!")
    print(f"Total records processed: {stats['total_records']}")
    print(f"Response records found: {stats['response_records']}")
    print(f"HTML pages extracted: {stats['html_pages']}")
    print(f"Other content types: {stats['other_content']}")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Your WARC file path
    warc_path = compressed_file
    
    print(f"Starting extraction from {warc_path}")
    extract_html_pages(warc_path)