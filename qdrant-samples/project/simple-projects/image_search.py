from PIL import Image
from transformers import AutoFeatureExtractor, AutoModel
import torch
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct



extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
model = AutoModel.from_pretrained("google/vit-base-patch16-224")


client = QdrantClient("localhost", port=6333)

client.create_collection(
    collection_name="images",
    vectors_config=VectorParams(size=768 , distance=Distance.COSINE)
)

from PIL import Image, UnidentifiedImageError

def get_image_embedding(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
    except UnidentifiedImageError:
        print(f"Cannot open image file: {image_path}")
        return None
    
    inputs = extractor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    embedding = outputs.last_hidden_state[:,0,:].numpy().flatten()
    return embedding

image_dir = "hello"
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

points = []
for i, image_file in enumerate(image_files):
    image_path = os.path.join(image_dir, image_file)
    embedding = get_image_embedding(image_path)
    points.append(
        PointStruct(
            id=i,
            vector=embedding.tolist(),
            payload={"file_name": image_file, "path": image_path}
        )
    )

client.upsert(collection_name="images", points=points)

def find_similar_images(query_image_path, top_k=5):
    query_embedding = get_image_embedding(query_image_path)
    results = client.search(
        collection_name="images",
        query_vector=query_embedding.tolist(),
        limit=top_k
    )
    return [(result.payload["file_name"], result.score) for result in results]


res = find_similar_images('ford.jpg')
print(res)