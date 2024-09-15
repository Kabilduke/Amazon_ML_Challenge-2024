import pandas as pd
from tqdm import tqdm
import os
import requests
import urllib
from PIL import Image
from functools import partial
import multiprocessing

def create_placeholder_image(image_save_path):
    try:
        placeholder_image = Image.new('RGB', (100, 100), color='black')
        placeholder_image.save(image_save_path)
    except Exception as e:
        print(f"Error creating placeholder image: {e}")

def download_image(image_link, save_folder, retries=3, delay=3):
    if not isinstance(image_link, str):
        print(f"Invalid image link: {image_link}")
        return

    filename = os.path.join(save_folder, os.path.basename(image_link))

    if os.path.exists(filename):
        print(f"Image already exists: {filename}")
        return

    for _ in range(retries):
        try:
            urllib.request.urlretrieve(image_link, filename)
            print(f"Downloaded: {filename}")
            return
        except Exception as e:
            print(f"Error downloading {image_link}: {e}")
            time.sleep(delay)
    

    print(f"Failed to download {image_link}, creating placeholder.")
    create_placeholder_image(filename)

def download_images(image_links, download_folder, allow_multiprocessing=True):
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    if allow_multiprocessing:
        download_image_partial = partial(download_image, save_folder=download_folder, retries=3, delay=3)

        with multiprocessing.Pool(4) as pool:  
            list(tqdm(pool.imap(download_image_partial, image_links), total=len(image_links)))
            pool.close()
            pool.join()
    else:
        for image_link in tqdm(image_links, total=len(image_links)):
            download_image(image_link, save_folder=download_folder, retries=3, delay=3)

# Load dataset and subset 10,000 rows
df = pd.read_csv("Amazon_train.csv") # Enter your path for the dataset(csv)
df_subset = df.head(10000)  # Use only the first 10,000 rows

image_links = df_subset['image_link'].tolist()
download_folder = "./Amazon_images" #Enter your folder path

download_images(image_links, download_folder, allow_multiprocessing=False)
