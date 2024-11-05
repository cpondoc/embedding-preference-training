# import the necessary functions
import requests
from trafilatura import fetch_url, extract, html2txt
from tqdm import tqdm
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
    urls, titles = [], []
    with open("data/wiki/good-articles.txt", "r", encoding="utf-8") as file:
        for line in file:
            url = line.split(" ")[0]
            page_url = f"https://en.wikipedia.org{url}"
            urls.append(page_url)
            titles.append(get_wikipedia_title_from_url(url))

    return urls, titles

def get_bad_articles(good_titles):
    """
    Get random Wikipedia articles.
    """
    urls, titles = [], set()
    while len(titles) < len(good_titles):
        new_rando, new_url = get_random_wikipedia_page()
        if new_rando not in good_titles:
            titles.add(new_rando)
            urls.append(new_url)
            
            with open("data/bad-wiki.txt", "a") as file:
                file.write(f"{new_url} - {new_rando}\n")
        print(len(good_titles) - len(titles))
    return urls, titles

def read_bad_articles():
    """
    Read from file.
    """
    urls, titles = [], []
    with open("data/bad-wiki.txt", "r", encoding="utf-8") as file:
        for line in file:
            url = line.split(" ")[0]
            urls.append(url)
            titles.append(get_wikipedia_title_from_url(url))

    return urls, titles
    
def save_to_file(content, article_type, title):
    """
    Save content to a text file.
    """
    with open("data/wiki-vs/" + article_type + "/" + title + ".txt" , "w", encoding="utf-8") as file:
        if content:
            file.write(content)

if __name__ == "__main__":
    good_urls, good_titles = get_good_articles()
    bad_urls, bad_titles = read_bad_articles()
    for i in tqdm(range(len(good_urls))):
        url, title = good_urls[i], good_titles[i]
        content = use_trafilatura(url)
        save_to_file(content, "good", title)
    
    for i in tqdm(range(len(bad_urls))):
        url, title = bad_urls[i], bad_titles[i]
        content = use_trafilatura(url)
        save_to_file(content, "bad", title)
    # bad_urls, bad_titles = get_bad_articles(good_titles)
