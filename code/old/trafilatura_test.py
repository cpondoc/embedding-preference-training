# import the necessary functions
from trafilatura import fetch_url, extract


def use_trafilatura(url):
    """
    Use Trafilatura.
    """
    # grab a HTML file to extract data from
    downloaded = fetch_url(url)

    # output main content and comments as plain text
    result = extract(downloaded)
    return result


with open("data/external-links.txt", "r", encoding="utf-8") as file:
    counter = 0
    for line in file:
        url = line.strip()

        # Extract content using Trafilatura
        content = use_trafilatura(url)
        output_file = "data/good/good-" + str(counter) + ".txt"

        # Save the extracted content if any
        if content:
            with open(output_file, "a", encoding="utf-8") as outfile:
                outfile.write(content)

        counter += 1
