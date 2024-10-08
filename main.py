import requests


def get_internal_links(page_title):
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


# Example usage
page_title = "Python_(programming_language)"
internal_links = get_internal_links(page_title)
external_links = get_external_links(page_title)

print(f"Internal Wikipedia Links ({len(internal_links)}):\n", internal_links[:10], "\n")
print(f"External Links ({len(external_links)}):\n", external_links[:10])
