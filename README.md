# README for Image Metadata Fetching and Processing Project

## Overview

This project consists of three Python scripts that work together to fetch image URLs using the SerpApi, extract metadata from these URLs, and process the data for further analysis. The primary goal is to facilitate the collection of image metadata from Google search results based on user-provided keywords.

## Files

1. **main.py**
2. **utils.py**
3. **serpapi_search.py**

## Requirements

- Python 3.x
- Required libraries:
  - `torch`
  - `datasets`
  - `transformers`
  - `pinecone-client`
  - `bs4` (BeautifulSoup)
  - `serpapi`
  - `json`
  - `re`
  - `csv`
  - `datetime`
  - `urllib`

You can install the required libraries using the following command:
```bash
pip install torch datasets transformers pinecone-client beautifulsoup4 serpapi
```

## File Descriptions

### 1. main.py

This is the main script that orchestrates the workflow. It prompts the user for keywords to search for images, fetches image URLs using the SerpApi, extracts metadata, and processes the data.

#### Functions

- `prompt_image_source()`: Prompts the user to choose between uploading a local image or providing an image URL.
- `prompt_keywords()`: Prompts the user to input keywords for searching images.
- `if __name__ == '__main__'`: The main entry point of the script. It fetches image URLs based on user-provided keywords, sets up the model, processes the images, and plots the top matches.

### 2. utils.py

This script contains utility functions used by `main.py` to handle model setup, image processing, and plotting.

#### Functions

- `get_model_info()`: Fetches the model, processor, and tokenizer based on the provided model ID.
- `get_single_image_embedding()`: Gets the embedding for a single image.
- `get_all_image_embeddings_from_urls()`: Gets embeddings for multiple images from URLs.
- `plot_top_matches_seaborn()`: Plots the top matching images using seaborn.
- `read_image_url()`: Reads an image from a URL.
- `read_local_image()`: Reads a local image.
- `create_data_to_upsert_from_urls()`: Creates data to upsert from URLs.

### 3. serpapi_search.py

This script handles the interaction with the SerpApi to fetch image URLs based on keywords.

#### Functions

- `fetch_metadata(url)`: Fetches metadata for a given URL.
- `count_urls(urls)`: Counts the number of unique URLs.
- `image_lookup(keywords)`: Searches for images using SerpApi based on provided keywords.
- `main()`: The main entry point of the script. It fetches image URLs based on user-provided keywords, extracts metadata, and saves the data to a JSON file.

## Usage

1. **Run the main script**:
   ```bash
   python main.py
   ```
2. **Follow the prompts**:
   - Enter keywords to search for images.
   - The script will fetch image URLs, extract metadata, and process the data.

## Output

The script generates two main output files:
- `image_urls.json`: Contains the extracted image URLs.
- `url_metadata.json`: Contains the metadata for each URL.

## Notes

- Ensure you have a valid SerpApi key and replace the placeholder in the `serpapi_search.py` script with your actual key.
- The script handles exceptions and prints error messages if metadata fetching fails for any URL.

This README provides a detailed overview of the project, helping users understand and utilize the code effectively.
