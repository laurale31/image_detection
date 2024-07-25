import torch
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from pinecone import Pinecone, ServerlessSpec
from utils import (
    get_model_info,
    get_single_image_embedding,
    get_all_image_embeddings_from_urls,
    plot_top_matches_seaborn,
    read_image_url,
    read_local_image
)
from serpapi_search import image_lookup

def prompt_image_source():
    while True:
        choice = input("Do you want to upload an image locally or provide an image URL? (local/url): ").strip().lower()
        if choice == 'local':
            file_path = input("Please enter the file path of the image: ").strip()
            return read_local_image(file_path)
        elif choice == 'url':
            image_url = input("Please enter the image URL: ").strip()
            return read_image_url(image_url)
        else:
            print("Invalid choice. Please enter 'local' or 'url'.")

def prompt_keywords():
    while True:
        keywords = input("Enter keywords to search images: ").strip()
        if keywords and not keywords.isspace():
            return keywords
        else:
            print("Invalid input. Please enter a valid keyword string.")


if __name__ == '__main__':

    # Fetch image URLs from Google search
   
    #list_image_urls= get_image_urls('lord of the rings film scenes')
    #list_image_urls=image_lookup('dune part two movie scenes')
    keywords = prompt_keywords()
    list_image_urls = image_lookup(keywords)


    # Set the device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define the model ID
    model_ID = "openai/clip-vit-base-patch32"

    # Get model, processor & tokenizer
    model, processor, tokenizer = get_model_info(model_ID, device)

    # Example usage: Get embeddings for multiple images
    img_embeddings, list_valid_image_urls = get_all_image_embeddings_from_urls(list_image_urls, processor, model, device, num_images=100)
    assert len(img_embeddings) == len(list_valid_image_urls)


    # Initialize Pinecone
    pc = Pinecone(api_key='API_KEY')
   

    # Define index name and dimension
    index_name = "duneimage2"
    vector_dim = 512  # Assuming embeddings are 1D arrays

    # Check if index exists
    if index_name not in pc.list_indexes().names():
        print("Index not present, creating index...")
        pc.create_index(
            name=index_name, 
            dimension=vector_dim, 
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )

    # Connect to the Pinecone index
    my_index = pc.Index(index_name)

    data_to_upsert = create_data_to_upsert_from_urls(list_valid_image_urls, img_embeddings, len(list_valid_image_urls))
    

    my_index.upsert(vectors = data_to_upsert)
    print("Pinecone upserted successfully")

    # Example usage: Get single image embedding
    image_query = prompt_image_source()
    query_embedding = get_single_image_embedding(image_query, processor, model, device)
    img_results = my_index.query(
        vector = query_embedding, 
        top_k=5, 
        include_metadata=True
        )

    # Example usage: Plot top matches using Seaborn
    plot_top_matches_seaborn(img_results)
