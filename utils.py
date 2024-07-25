import numpy as np
from PIL import Image
from io import BytesIO
import requests
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer

def read_image_url(image_URL):
   response = requests.get(image_URL)
   image = Image.open(BytesIO(response.content)).convert("RGB")
   image.show()
   return image

def read_local_image(file_path):
    try:
        # Open an image file
        with Image.open(file_path) as img:
            # Convert the image to RGB
            rgb_img = img.convert("RGB")
            # Show the image
            rgb_img.show()
            return rgb_img
    except IOError:
        print("Error: Unable to open image file.")

def get_model_info(model_ID, device):
    # Save the model to device
    model = CLIPModel.from_pretrained(model_ID).to(device)
    # Get the processor
    processor = CLIPProcessor.from_pretrained(model_ID)
    # Get the tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(model_ID)
    return model, processor, tokenizer

def get_single_image_embedding(my_image, processor, model, device):
    image = processor(
        text=None,
        images=my_image,
        return_tensors="pt"
    )["pixel_values"].to(device)
    embedding = model.get_image_features(image)
    return embedding.cpu().detach().numpy().flatten().tolist()

def check_valid_URL(image_URL):
    try:
        response = requests.get(image_URL)
        if response.status_code != 200:
            return False
        Image.open(BytesIO(response.content))
        return True
    except Exception:
        return False

def get_all_image_embeddings_from_urls(dataset, processor, model, device, num_images=100):
    embeddings = []

    # Limit the number of images to process
    dataset = dataset[:num_images]
    working_urls = []

    #for image_url in dataset['image_url']:
    for image_url in dataset:
      if check_valid_URL(image_url):
          try:
              # Download the image
              response = requests.get(image_url)
              image = Image.open(BytesIO(response.content)).convert("RGB")
              # Get the embedding for the image
              embedding = get_single_image_embedding(image, processor, model, device)
              #embedding = get_single_image_embedding(image)
              embeddings.append(embedding)
              working_urls.append(image_url)
          except Exception as e:
              print(f"Error processing image from {image_url}: {e}")
      else:
          print(f"Invalid or inaccessible image URL: {image_url}")

    return embeddings, working_urls

def create_data_to_upsert_from_urls(dataset,  embeddings, num_images):
  metadata = []
  image_IDs = []
  for index in range(len(dataset)):
    metadata.append({
        'ID': index,
        'image': dataset[index]#['image_url']
    })
    image_IDs.append(str(index))

  image_embeddings = [arr for arr in embeddings]
  data_to_upsert = list(zip( image_IDs, image_embeddings, metadata))
  return data_to_upsert

def extract_highest_score(match_data):
  # Ensure there are matches in the input data
  if 'matches' in match_data and isinstance(match_data['matches'], list) and len(match_data['matches']) > 0:
      # Find the match with the highest score
      highest_score_match = max(match_data['matches'], key=lambda x: x['score'])

      # Extract the required keys for the highest score
      result = {
          'id': highest_score_match['id'],
          'score': highest_score_match['score'],
          'image_url': highest_score_match['metadata']['image']
      }
      return result
  else:
      return None

def plot_top_matches_seaborn(match_data):
    if 'matches' in match_data and isinstance(match_data['matches'], list) and len(match_data['matches']) > 0:
        # Sort the matches by score in descending order and take the top 3
        sorted_matches = sorted(match_data['matches'], key=lambda x: x['score'], reverse=True)[:5]

        # Create a grid for displaying the top 3 matches
        fig, axs = plt.subplots(1, 5, figsize=(25, 5))

        for i, match in enumerate(sorted_matches):
            ax = axs[i]
            image_url = match['metadata']['image']
            id_val = match['id']
            score_val = match['score']

            try:
                # Load and display the image using Seaborn
                response = requests.get(image_url)
                response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
                img = Image.open(BytesIO(response.content))
                img = img.resize((300, 300))
                ax.imshow(img)
                ax.set_title(f"ID: {id_val}\nScore: {score_val:.2f}")
                ax.axis('off')
            except requests.exceptions.RequestException as e:
                print(f"Error fetching image from URL {image_url}: {e}")
                ax.set_title(f"ID: {id_val}\nScore: {score_val:.2f}\nImage not available")
                ax.axis('off')

        plt.show()
    else:
        print("No 'matches' found in the input data or the list is empty.")

