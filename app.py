from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from PIL import Image
from io import BytesIO
import numpy as np
#import tensorflow as tf
from scipy.spatial.distance import cosine
import pickle

app = FastAPI()

# Load the pre-trained model
#model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

# Load precomputed features
with open('product_features.pkl', 'rb') as f:
    product_database = pickle.load(f)

def preprocess_image(image):
    image = image.resize((224, 224)).convert('RGB')
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

def extract_features(image):
    img_array = preprocess_image(image)
    return model.predict(img_array).flatten()

def find_similar_products(query_features):
    similarities = [
        (product_id, 1 - cosine(query_features, data["features"]))
        for product_id, data in product_database.items()
    ]
    sorted_products = sorted(similarities, key=lambda x: x[1], reverse=True)[:5]
    return [{"product_id": product_id, "img_link": product_database[product_id]["img_link"]} for product_id, _ in sorted_products]

class ImageRequest(BaseModel):
    img_url: str

@app.post("/recommend")
async def recommend_image(request: ImageRequest):
    try:
        response = requests.get(request.img_url)
        image = Image.open(BytesIO(response.content))
        query_features = extract_features(image)
        recommendations = find_similar_products(query_features)
        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))