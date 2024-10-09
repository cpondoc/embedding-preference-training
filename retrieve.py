import requests
from bs4 import BeautifulSoup

URL = "https://en.wikipedia.org/wiki/Wikipedia:Good_articles/all"
ARTICLE_NAMES_FILE_NAME = "data/good-articles.txt"


def get_internal_links(page_title):
    """
    Calling the Mediawiki API in order to get all internal links.
    """
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",  # The action to perform (query)
        "prop": "links",  # Get internal links (Wikipedia links)
        "titles": page_title,  # The title of the Wikipedia page
        "pllimit": "max",  # Limit of internal links to fetch (max is 500)
        "format": "json",
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        pages = data.get("query", {}).get("pages", {})
        links = []
        for page in pages.values():
            links.extend(link["title"] for link in page.get("links", []))
        return links
    return []


def get_external_links(page_title):
    """
    Calling the Mediawiki API in order to get all external links.
    """
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",  # The action to perform (query)
        "prop": "extlinks",  # Get external links (external websites)
        "titles": page_title,  # The title of the Wikipedia page
        "ellimit": "max",  # Limit of external links to fetch (max is 500)
        "format": "json",
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        pages = data.get("query", {}).get("pages", {})
        external_links = []
        for page in pages.values():
            external_links.extend(link["*"] for link in page.get("extlinks", []))
        return external_links
    return []


def retrieve_good_articles():
    """
    Retrieve list of good articles, save them locally
    """
    # Make GET request to endpoint
    response = requests.get(URL)
    soup = BeautifulSoup(response.content, "html.parser")
    collapsible_contents = soup.find_all(class_="mw-collapsible-content")

    # Save text file and append
    with open(ARTICLE_NAMES_FILE_NAME, "a") as file:
        for section in collapsible_contents:
            a_tags = section.find_all("a")
            for a in a_tags:
                file.write(f"{a['href']} - {a.text}\n")


def extract_links():
    """
    Scrape file and save all the appropriate links.
    """
    with open(ARTICLE_NAMES_FILE_NAME, "r") as file:
        # Get the specific query
        for line in file:
            arr = line.split(" ")
            query = arr[0][6:]

            # Save all external links
            with open("data/external-links.txt", "a") as ext_file:
                ext_links = get_external_links(query)
                for ext_link in ext_links:
                    ext_file.write(ext_link + "\n")

            # Save all internal links
            with open("data/internal-links.txt", "a") as int_file:
                int_links = get_internal_links(query)
                for int_link in int_links:
                    int_file.write(int_link + "\n")


if __name__ == "__main__":
    extract_links()
