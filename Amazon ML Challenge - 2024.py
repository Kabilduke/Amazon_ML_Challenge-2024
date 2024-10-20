#!/usr/bin/env python
# coding: utf-8

# # Amazon ML Challenge - 2024

# In[306]:


conda update --all


# In[196]:


import pandas as pd
import numpy as np


# In[197]:


df = pd.read_csv("Amazon_train.csv")
df.head(10)


# In[198]:


df_test = pd.read_csv("Amazon_test.csv")
df_test.head(10)


# In[199]:


df_test.shape


# In[8]:


df.info()


# In[11]:


df.shape


# In[22]:


df['entity_name'].value_counts()


# ### Downloading 10,000 only image from throught pipline using links

# In[37]:


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
    
    # If all retries fail, create a placeholder image
    print(f"Failed to download {image_link}, creating placeholder.")
    create_placeholder_image(filename)

def download_images(image_links, download_folder, allow_multiprocessing=True):
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    if allow_multiprocessing:
        download_image_partial = partial(download_image, save_folder=download_folder, retries=3, delay=3)

        with multiprocessing.Pool(4) as pool:  # Use fewer processes initially for debugging
            list(tqdm(pool.imap(download_image_partial, image_links), total=len(image_links)))
            pool.close()
            pool.join()
    else:
        for image_link in tqdm(image_links, total=len(image_links)):
            download_image(image_link, save_folder=download_folder, retries=3, delay=3)

# Load dataset and subset 10,000 rows
df = pd.read_csv("Amazon_train.csv")
df_subset = df.head(10000)  # Use only the first 10,000 rows

# Extract image links
image_links = df_subset['image_link'].tolist()

# Set folder to save images
download_folder = "./Amazon_images"

# Try without multiprocessing first to debug
download_images(image_links, download_folder, allow_multiprocessing=False)


# In[108]:


print(df["image_link"].head(10))


# In[190]:


import cv2
from PIL import Image
import matplotlib.pyplot as plt

image_data = "Amazon_images/71gSRbyXmoL.jpg"
image = cv2.imread(image_data)

image_r = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(image_grey, 50, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

thresh_image = Image.fromarray(thresh)

fig, axes = plt.subplots(1, 3, figsize=(20, 10))

axes[0].imshow(image_r)
axes[0].set_title("Original Image")

axes[1].imshow(image_grey)
axes[1].set_title("Gray scaled image")

axes[2].imshow(thresh_image)
axes[2].set_title("After Threshold")

plt.tight_layout()
plt.show()


# In[36]:


print(thresh)


# In[201]:


import os

url =  "https://m.media-amazon.com/images/I/1yw53vfQtS.jpg"
file_name = os.path.basename(url)
print(file_name)


# In[202]:


df["image_id"] = df["image_link"].apply(os.path.basename)
print(df[['image_link', 'image_id']].head())


# In[204]:


df.head(25)


# In[ ]:





# ## Tesseract OCR by Google

# In[85]:


pip install pytesseract


# In[ ]:





# In[347]:


image_data = "Amazon_images/91-iahVGEDL.jpg"
image = cv2.imread(image_data)

image_r = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image_r)
plt.title("Original image")
plt.axis("off")
plt.show()


# In[348]:


from PIL import Image
import re

def process_image(image_url):
    img = cv2.imread(image_url)
    
    image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshs = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Four paremeters *gray_image, minimum threshold, maximium threshold, type of threshold*.
    thresh_img = Image.fromarray(threshs)
    return thresh_img
    
def extract_text(image):
    pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(image, lang = 'eng', config= custom_config)
    return text

def clean_text(text):
    cleaned_text = re.sub(r'\s+', ' ', text).strip()
    return cleaned_text

image_url = 'Amazon_images/91-iahVGEDL.jpg'
processed_image = process_image(image_url)
text = extract_text(processed_image)
cleaned_text = clean_text(text)
print(cleaned_text) 


# In[349]:


import pytesseract
import cv2

image = cv2.imread("Amazon_images/91-iahVGEDL.jpg")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
custom_config = r'--oem 3 --psm 6'
text = pytesseract.image_to_string(gray_image, lang = 'eng', config=custom_config)
print(text)


# ## Conculsion -- Score 70%

# # --------------------------------------------------------------------------------------------------------------------

# # EasyOCR

# In[140]:


pip install easyocr


# In[351]:


import easyocr
import cv2
import matplotlib.pyplot as plt

reader = easyocr.Reader(['en'])

image_way = 'Amazon_images/91-iahVGEDL.jpg'
image_w = cv2.imread(image_way)

image_rgb = cv2.cvtColor(image_w, cv2.COLOR_BGR2GRAY)

results = reader.readtext(image_rgb)

for result in results:
    print(f"Detected text: {result[1]} (Confidence: {result[2]:.2f})")


# In[352]:


#Store the result in a string for NER.
detected_text = ""

for extract in results:
    text = extract[1]
    detected_text += text + " "
    
detected_text = detected_text.strip()
print(detected_text)


# ## Conclusion - Score 95%

# In[ ]:





# # Named Entity Regconition (NER)

# In[323]:


detected_text


# In[214]:


pip install spacy


# In[220]:


get_ipython().system('python -m spacy download en_core_web_sm')


# In[353]:


import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp(detected_text)

nlp.pipe_names


# In[354]:


for ent in doc.ents:
    print(f"Entity: {ent.text}, Label: {spacy.explain(ent.label_)}")


# In[355]:


from spacy import displacy

displacy.render(doc, style="ent")


# In[356]:


nlp.pipe_labels['ner']


# In[358]:


import re

expression = detected_text
pattern = r'(\d+[\.\d]*)([A-za-a]+)'

matches = re.findall(pattern, expression)

print(matches)


# In[ ]:





# In[ ]:




