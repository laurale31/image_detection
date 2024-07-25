from serpapi import GoogleSearch
import json
from datetime import datetime
import urllib.request
from http.cookiejar import CookieJar
from bs4 import BeautifulSoup

# Setting up the cookie jar and opener
cj = CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
opener.addheaders = [
    ('User-agent', 'Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.27 Safari/537.17')
]

def fetch_metadata(url):
    """Fetch metadata for a given URL."""
    try:
        response = opener.open(url)
        html = response.read().decode('utf-8')
        soup = BeautifulSoup(html, 'html.parser')

        metadata = {
            'title': soup.title.string if soup.title else 'No title',
            'description': '',
            'keywords': '',
            'time': None,  # None to denote no time found initially
            'location': ''
        }

        for meta in soup.find_all('meta'):
            if 'name' in meta.attrs:
                if meta.attrs['name'].lower() == 'description':
                    metadata['description'] = meta.attrs['content']
                if meta.attrs['name'].lower() == 'keywords':
                    metadata['keywords'] = meta.attrs['content']
            if 'property' in meta.attrs:
                if meta.attrs['property'].lower() in ['og:time', 'article:published_time', 'article:modified_time']:
                    metadata['time'] = meta.attrs['content']
                if meta.attrs['property'].lower() in ['og:location', 'article:section']:
                    metadata['location'] = meta.attrs['content']

        # Additional heuristic to search for time information within <time> tags or text
        if metadata['time'] is None:
            time_tags = soup.find_all('time')
            if time_tags:
                metadata['time'] = time_tags[0].get('datetime', time_tags[0].text)

        # Convert the time to a standard format, if available
        if metadata['time']:
            try:
                metadata['time'] = datetime.fromisoformat(metadata['time']).isoformat()
            except ValueError:
                metadata['time'] = str(datetime.now())  # Fallback to current time if parsing fails

        return metadata
    except Exception as e:
        print(f"Failed to fetch metadata for {url}: {e}")
        return None

def count_urls(urls):
    """Count the number of unique URLs."""
    return len(urls)

def image_lookup(keywords):
    # SerpApi search parameters
    params = {
        "engine": "google",
        "q": keywords,
        "tbm": "isch",  # Image search
        "num": 100,
        "api_key": "SERPAPI_KEY"  # Replace with your actual SerpApi key
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    organic_results = results.get("images_results", [])

    # Extract image URLs
    image_urls = [result['original'] for result in organic_results if 'original' in result]

    # Dump image URLs to a JSON file
    with open('image_urls.json', 'w') as json_file:
        json.dump(image_urls, json_file, indent=2)

    return image_urls

def main():
    keywords = input("Enter keywords to search images: ")
    urls = image_lookup(keywords)
    print("Extracted URLs:", urls)
    num_urls = count_urls(urls)
    print(f"Number of unique URLs: {num_urls}")

    # Store metadata for each unique URL
    url_metadata = {}
    for url in urls:
        metadata = fetch_metadata(url)
        if metadata:
            url_metadata[url] = metadata

    # Dump metadata to a JSON file
    with open('url_metadata.json', 'w') as json_file:
        json.dump(url_metadata, json_file, indent=4)

if __name__ == "__main__":
    main()
