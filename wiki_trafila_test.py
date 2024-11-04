# import the necessary functions
import requests
from trafilatura import fetch_url, extract
from urllib.parse import unquote


def get_wikipedia_title_from_url(url):
    # Split the URL by '/' and get the last part, which is the title
    title = url.split("/")[-1]
    # Decode any URL-encoded characters (e.g., underscores to spaces)
    title = unquote(title).replace("_", " ")
    return title


def use_trafilatura(url):
    """
    Use Trafilatura.
    """
    # grab a HTML file to extract data from
    downloaded = fetch_url(url)

    # output main content and comments as plain text
    result = extract(downloaded)
    return result


def get_random_wikipedia_page():
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

    pages = data["query"]["pages"]
    page = next(iter(pages.values()))  # Get the first (random) page
    title = page.get("title", "No title")

    # Build the full URL to the Wikipedia page
    page_url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"

    return title, page_url


def get_good_articles():
    """
    Get all of the good Wikipedia articles.
    """
    urls, titles = [], set()
    with open("data/wiki/good-articles.txt", "r", encoding="utf-8") as file:
        for line in file:
            url = line.split(" ")[0]
            page_url = f"https://en.wikipedia.org{url}"
            urls.append(page_url)
            titles.add(get_wikipedia_title_from_url(url))

    return urls, titles


if __name__ == "__main__":
    urls, titles = get_good_articles()
    bad_urls, bad_titles = [], set()
    while len(bad_titles) < len(titles):
        new_rando, new_url = get_random_wikipedia_page()
        if new_rando not in titles:
            bad_titles.add(new_rando)
            bad_urls.append(new_url)
        print(new_url)
    print(bad_titles)
