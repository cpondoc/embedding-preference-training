"""
Data for scraping good and bad Wikipedia articles.
"""

from tqdm import tqdm
from helpers import *


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


def find_bad_articles(good_titles):
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


def get_bad_articles():
    """
    Read from file of bad articles.
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
    with open(
        "data/wiki-vsd/" + article_type + "/" + title + ".txt", "w", encoding="utf-8"
    ) as file:
        if content:
            file.write(content)


def main():
    """
    Load in good and bad articles, and save them to file.
    """
    # Load in good and bad articles
    good_urls, good_titles = get_good_articles()
    bad_urls, bad_titles = get_bad_articles()

    # Extract good articles, save to file
    for i in tqdm(range(len(good_urls))):
        url, title = good_urls[i], good_titles[i]
        content = use_trafilatura(url)
        save_to_file(content, "good", title)

    # Extract bad articles, save to file
    for i in tqdm(range(len(bad_urls))):
        url, title = bad_urls[i], bad_titles[i]
        content = use_trafilatura(url)
        save_to_file(content, "bad", title)


if __name__ == "__main__":
    main()
