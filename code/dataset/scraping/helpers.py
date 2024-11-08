"""
General helpers to assist with web scraping + parsing Wikipedia.
"""

import requests
from trafilatura import fetch_url, extract, html2txt
from tqdm import tqdm
from urllib.parse import unquote


def get_wikipedia_title_from_url(url):
    """
    General helper function to get Wikipedia title from a URL.
    """
    title = url.split("/")[-1]
    title = unquote(title).replace("_", " ")
    return title


def use_trafilatura(url):
    """
    Use Trafilatura in order to be able to extract contents from webpage.
    """
    downloaded = fetch_url(url)
    result = extract(downloaded)
    return result


def get_random_wikipedia_page():
    """
    Use the Wikipedia API to get random Wikipedia pages.
    """
    # Form URL and make request
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "generator": "random",
        "grnnamespace": 0,  # 0 for articles, excluding talk pages, etc.
        "prop": "extracts",  # Get the content summary
        "explaintext": True,  # Plain text format
        "exintro": True,  # Only fetch the introduction
    }
    response = requests.get(url, params=params)
    data = response.json()

    # Get the first (random) page
    pages = data["query"]["pages"]
    page = next(iter(pages.values()))
    title = page.get("title", "No title")

    # Build the full URL to the Wikipedia page
    page_url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
    return title, page_url
