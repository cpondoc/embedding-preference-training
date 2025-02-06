"""
Code to extract WARCs and use a random subset of them.
"""

import fasttext
import tarfile
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
from tqdm import tqdm
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


def extract_tar_gz_files(blocklist_root, extract_dir):
    """
    Recursively extract all .tar.gz files to a specific directory.
    """
    for root, dirs, files in os.walk(blocklist_root):
        for file in files:
            if file.endswith(".tar.gz"):
                tar_gz_path = os.path.join(root, file)
                with tarfile.open(tar_gz_path, "r:gz") as tar:
                    tar.extractall(path=extract_dir)
                print(f"Extracted: {tar_gz_path}")


def download_file(url, output_path):
    """
    Download a zipped file if it doesn't already exist.
    """
    # If file exists, return output path
    if os.path.exists(output_path):
        print(f"File already exists: {output_path}")
        return output_path

    # If file does not exist, download
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


def fasttext_english_filter(content: str):
    """
    Using similar Fineweb techniques to just filter out some noise.
    """
    # Define pre-trained fasttext model
    model_path = "models/lid.176.bin"
    model = fasttext.load_model(model_path)

    # Inference, and return (can tune 0.6 probability)
    label, probability = model.predict(content, k=1)
    if label[0] == "__label__en" and probability[0] > 0.6:
        return True
    return False


def load_blocklist(blocklist_root):
    """
    Load blocklisted domains and URLs from hierarchical folders.
    """
    blocked_domains = set()
    blocked_urls = set()

    # Traverse blocklist folders (e.g., 'adult', 'ads', etc.)
    for category in os.listdir(blocklist_root):
        category_path = os.path.join(blocklist_root, category)

        if not os.path.isdir(category_path):
            continue  # Skip non-directory files

        # Load domains
        domain_file = os.path.join(category_path, "domains")
        if os.path.exists(domain_file):
            with open(domain_file, "r", encoding="utf-8") as f:
                blocked_domains.update(line.strip() for line in f if line.strip())

        # Load URLs
        url_file = os.path.join(category_path, "URLs")
        if os.path.exists(url_file):
            with open(url_file, "r", encoding="utf-8") as f:
                blocked_urls.update(line.strip() for line in f if line.strip())

    return blocked_domains, blocked_urls


def is_blocked(url, blocked_domains, blocked_urls):
    """
    Check if a given URL is blocklisted based on domains or URLs.
    """
    parsed_url = urlparse(url)
    domain = parsed_url.netloc

    return domain in blocked_domains or url in blocked_urls


def save_to_hf(warc_file_name, output_dir="data/noisy-cc"):
    """
    Save to HuggingFace + delete file
    """
    dataset = load_dataset("csv", data_files=f"{output_dir}/{warc_file_name}.csv")
    dataset.push_to_hub("cpondoc/noisy-cc")
    os.remove(f"{output_dir}/{warc_file_name}.csv")


def extract_html_pages(
    warc_path, blocked_domains, blocked_urls, warc_file_name, output_dir="data/noisy-cc"
):
    """
    Extracting HTML pages and running data validation pipeline.
    """

    # Create output directory and CSV file name
    os.makedirs(output_dir, exist_ok=True)
    data = []

    # [TO-DO] Check Fineweb URLs
    # all_urls = get_fineweb_urls()
    # all_urls = set()

    # Open the WARC, iterate through each record
    try:
        with open(warc_path, "rb") as stream:
            for record in tqdm(ArchiveIterator(stream), desc="Processing WARC records"):

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
                        if (
                            content
                            and fasttext_english_filter(content)
                            and not is_blocked(
                                target_uri, blocked_domains, blocked_urls
                            )
                        ):
                            data.append({"url": target_uri, "text": content})

                    except Exception as e:
                        # logging.error(f"Error processing text: {e}")
                        continue

    except Exception as e:
        logging.error(f"Error processing WARC file: {e}")

    # Save to a CSV file
    df = pd.DataFrame(data)
    df.to_csv(f"{output_dir}/{warc_file_name}.csv", index=False)


if __name__ == "__main__":
    """
    Run all larger operations
    """

    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Process path of WARC into usable things.
    indices = get_warc_indices()
    for index in range(1, 6):
        path = indices[index]

        # Downloading file
        warc_paths_url = "https://data.commoncrawl.org/" + path
        compressed_file = "data/common-crawl/" + path.split("/")[-1]
        download_file(warc_paths_url, compressed_file)
        print(f"Loaded in WARC file: {warc_paths_url}.")

        # Set up blocklist
        # extract_tar_gz_files("data/blocklist/", "data/blocklist-unzip") # Run if we need to unzip blocklist
        blocked_domains, blocked_urls = load_blocklist("data/blocklist-unzip")
        print("Defined the blocklist.")

        # Extract HTML pages
        warc_path = compressed_file
        warc_file_name = warc_path.split("/")[-1]
        warc_file_name = warc_file_name.split(".")[0]
        print(f"Starting extraction from {warc_path}")
        extract_html_pages(warc_path, blocked_domains, blocked_urls, warc_file_name)

        # Save to HuggingFace + delete file
        save_to_hf(warc_file_name)
        if os.path.exists(compressed_file):
            os.remove(compressed_file)
