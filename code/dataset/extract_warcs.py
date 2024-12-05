"""
Code to extract WARCs and use a random subset of them.
"""

import requests
import gzip
import shutil
import os
from warcio.archiveiterator import ArchiveIterator
from urllib.parse import urlparse
from io import BytesIO
import trafilatura
import langdetect
import logging
from datetime import datetime
import re
from collections import defaultdict
from datasets import load_dataset
from tqdm import tqdm


# Function to download a file
def download_file(url, output_path):
    """
    Download a zipped file.
    """
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(output_path, "wb") as f:
            shutil.copyfileobj(response.raw, f)
        print(f"Downloaded: {output_path}")
    else:
        print(f"Failed to download {url}: {response.status_code}")
    return output_path


# Function to decompress a .gz file
def decompress_gz(file_path, output_path):
    """
    Function to decompress gz.
    """
    print(f"Decompressing {file_path}...")
    with gzip.open(file_path, "rb") as f_in:
        with open(output_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"Decompressed to: {output_path}")
    return output_path


def get_warc_indices():
    """
    Retrieve all the WARC paths.
    """
    # Loop through each line in the file, which contains WARC paths
    indices = []
    with open("data/common-crawl/warc.paths.txt", "r") as file:
        for line in file:

            # Process each line (you can modify this as needed)
            line = line.strip()
            if line[-3:] == ".gz":
                indices.append(line)
    return indices


# Now download and compress
# download_file(warc_paths_url, compressed_file)


def clean_filename(url):
    """
    Create a safe filename from URL.
    """
    # Remove protocol and special characters
    name = re.sub(r"[^\w\-_.]", "_", url)
    return name[:150] + ".txt"


def is_english(text):
    """
    Use LangDetect to use only English sources.
    """
    try:
        return langdetect.detect(text) == "en"
    except:
        return False


def get_fineweb_urls():
    """
    Get all URLs for Fineweb.
    """
    fw = load_dataset(
        "HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True
    )
    all_urls = set()
    for example in tqdm(fw, desc="Processing Fineweb URLs"):
        url = example["url"]
    all_urls.add(url)
    return all_urls


def extract_html_pages(warc_path, output_dir="data/random-cc"):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    all_urls = get_fineweb_urls()

    # Open the WARC, iterate through each record
    try:
        with open(warc_path, "rb") as stream:
            for record in ArchiveIterator(stream):

                # Check if it's a response record
                if record.rec_headers.get_header("WARC-Type") == "response":
                    target_uri = record.rec_headers.get_header("WARC-Target-URI")

                    # Read the payload
                    try:
                        payload = record.content_stream()
                        payload_text = payload.read().decode("utf-8", errors="ignore")

                        # Split out the body if it's an HTTP response
                        if "\r\n\r\n" in payload_text:
                            _, body = payload_text.split("\r\n\r\n", 1)
                        else:
                            body = payload_text

                        # Use trafilatura to extract main content, check if English
                        content = trafilatura.extract(body)
                        if content and is_english(content):

                            # Save to a file
                            if target_uri not in all_urls:
                                filename = clean_filename(target_uri)
                                filepath = os.path.join(output_dir, filename)
                                with open(filepath, "w", encoding="utf-8") as f:
                                    f.write(content)

                    except Exception as e:
                        logging.error(f"Error processing text: {e}")
                        continue

    except Exception as e:
        logging.error(f"Error processing WARC file: {e}")


if __name__ == "__main__":

    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Process path of WARC into usable things.
    indices = get_warc_indices()
    path = indices[0]
    warc_paths_url = "https://data.commoncrawl.org/" + path
    compressed_file = "data/common-crawl/" + path.split("/")[-1]
    warc_path = compressed_file

    # Extract HTML pages
    print(f"Starting extraction from {warc_path}")
    extract_html_pages(warc_path)
