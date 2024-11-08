"""
Code to utilize Common Crawl.
"""

from collections import deque
from warcio.archiveiterator import ArchiveIterator
import gzip
from bs4 import BeautifulSoup
from readability import Document
import json
import os
import requests
import shutil
from code.dataset.wiki import *

FILE_NAME = "data/common-crawl/cc-index.paths.gz"
TEXT_FILE = "data/common-crawl/cc-index.txt"
INDICES_FOLDER = "data/common-crawl/cc-indices"
LINKS_FOLDER = "data/common-crawl/cc-links"


def unzip_index(zip_name, file_name):
    """
    Unzips the core file name.
    """
    with gzip.open(zip_name, "rb") as f_in:
        with open(file_name, "wb") as f_out:
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
        with open("data/common-crawl/cc-pages" + "/" + file_name, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"File downloaded successfully as {file_name}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")


def read_index_file():
    """
    Read in all of the files, get all of the .gz files one by one
    """
    # Extract all lines from the .txt file
    with open(TEXT_FILE, "r") as file:
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
    wiki_set, uniq_ext = stats_on_links(EXTERNAL_LINKS_FILE)
    total_links = 0
    files = os.listdir(LINKS_FOLDER)
    for file in files:

        # Open said file, enumerate through each line
        file_name = LINKS_FOLDER + "/" + file
        with open(file_name, "r") as f:
            for _, line in enumerate(f):

                # Massage data
                line_arr = deque(line.split(" "))
                line_arr.popleft()
                line_arr.popleft()
                new_line = " ".join(line_arr)

                # First, check if we have an english language
                data = json.loads(new_line)
                if "languages" in data:
                    languages = data["languages"].split(",")
                    if "eng" in languages:

                        # Only add to file if we know it's not in the wiki set
                        url, filename = str(data["url"]), str(data["filename"])
                        if url not in wiki_set:
                            with open("data/common-crawl/cc-final.txt", "a") as new_f:
                                new_f.write(url + "," + filename + "\n")
                            total_links += 1

                            # Exit if we have enough links!
                            if total_links == uniq_ext:
                                return


def download_pages():
    """
    Download pages from Common Crawl file.
    """
    with open("data/common-crawl/cc-final.txt", "r") as f:
        count = 0
        for _, line in enumerate(f):
            crawl_path = line.strip().split(",")[1]
            obtain_individual_index(crawl_path)
            unzip_index(
                "data/common-crawl/cc-pages/" + crawl_path[-12:],
                "data/common-crawl/cc-txt/" + str(count) + ".txt",
            )
            count += 1
            if count == 100:
                return


def extract_text_from_warc(warc_gz_file):
    # Open the compressed WARC file
    count = 0
    with gzip.open(warc_gz_file, "rb") as stream:
        for record in ArchiveIterator(stream):
            # Only process 'response' records (which contain web content)
            if record.rec_type == "response":
                # Get the URL of the resource
                url = record.rec_headers.get_header("WARC-Target-URI")
                # Extract the HTML content
                content = record.content_stream().read()

                try:
                    # Use BeautifulSoup to parse the HTML
                    soup = BeautifulSoup(content, "lxml")
                    # Optionally use Readability to extract the main article content
                    doc = Document(str(soup))
                    main_html = doc.summary()  # Get the main content
                    main_text = BeautifulSoup(
                        main_html, "lxml"
                    ).get_text()  # Extract plain text

                    # Print the URL and the core text (or save to file)
                    # print(f"URL: {url}\nMain Text:\n{main_text}\n\n")
                    # print(f"URL: {url}")
                    # Alternatively, save content to a file if needed:
                    with open(
                        f"data/common-crawl/sample-data/{str(count)}.txt", "w+"
                    ) as output_file:
                        count += 1
                        output_file.write(f"{main_text}")
                        print(url)
                        print(count)

                    if count > 1000000:
                        return

                except Exception as e:
                    pass
                    # print(f"Error processing {url}: {e}")


if __name__ == "__main__":
    extract_text_from_warc("data/common-crawl/cc-pages/0262.warc.gz")
    # download_pages()
