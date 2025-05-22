from PIL import Image , UnidentifiedImageError
from transformers import AutoFeatureExtractor , AutoModel
import os 
import torch 
from sentence_transformers import SentenceTransformer
from help import Animal , count_images , cat_description

from qdrant_client import QdrantClient
from qdrant_client.models import Distance , VectorParams , PointStruct



extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
model = AutoModel.from_pretrained("google/vit-base-patch16-224")
text_model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight model for embeddings

client = QdrantClient('localhost' , port=6333)


list_colectio = client.collection_exists(collection_name="advanced_image_search")
print(list_colectio)    


def is_collection_exists(collection_name):
    try:
        res = client.collection_exists(collection_name=collection_name)
        if res:
            print(f"Collection '{collection_name}' exists.")
            return True
        else:
            print(f"Collection '{collection_name}' does not exist.")
            return False
    except Exception as e:
        print(f"Error checking collection existence: {e}")
        return False

def create_collection(collection_name):
    try:
        if not is_collection_exists(collection_name=collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "text": VectorParams(size=384, distance=Distance.COSINE),
                    "image": VectorParams(size=768, distance=Distance.COSINE)
                }
            )
            print(f"Collection '{collection_name}' created.")
        else: 
            print(f"Collection '{collection_name}' already exists.")
    except Exception as e:
        print(f"Error creating collection: {e}")

def embedding_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
    except UnidentifiedImageError:
        print(f"Cannot open image file: {image_path}")
        return None
    feature = extractor(images=img , return_tensors="pt")
    with torch.no_grad():
        outputs = model(**feature)

    embedding = outputs.last_hidden_state[:,0,:].numpy().flatten()
    return embedding    

def embedding_text(text):
    try:
        embedding = text_model.encode(text)
        return embedding
    except Exception as e:
        print(f"Error generating text embedding: {e}")
        return None

def get_points(
        id, 
        embedding_image,
        embedding_text, 
        image_path,
        file_name,
        category, 
        description,
        type_of_category
        ):
    try:
        points = PointStruct(
            id=id,
            vector={
                "image": embedding_image.tolist(),
                "text": embedding_text.tolist()
            },
            payload={
                "file_name": file_name,
                "path": image_path,
                "category": category,
                "description": description,
                "type_of_category": type_of_category
            }
        )
        return points
    except Exception as e:
        print(f"Error creating PointStruct: {e}")
        return None

def save_image_to_qdrant(collection_name , points , image_path):
    try:
        if not is_collection_exists(collection_name=collection_name):
            create_collection(collection_name=collection_name)
        if points is not None:
            client.upsert(
                collection_name=collection_name,
                points=points
            )
            print(f"Image '{image_path}' saved to Qdrant.")
        else:
            print(f"Failed to save image '{image_path}' to Qdrant: points is None.")
    except Exception as e:
        print(f"Error saving image to Qdrant: {e}")


def search_image_by_query_user(user_query, collection_name , top_k=5):
    try:
        query_embedding = embedding_text(user_query)
        results = client.search(
            collection_name=collection_name,
            query_vector={
                "text": query_embedding.tolist()
            },
            limit=top_k,
            filter=None
        )
        return [(result.payload["file_name"], result.score) for result in results]
    except Exception as e:
        print(f"Error searching image by user query: {e}")
        return None

def search_image_by_text_payload(
        category,
        description,
        type_of_category,
        colletion_name,
        query_text,
        top_k=5
        ):
    try:
        full_text = f"{category} {description} {type_of_category}"
        query_embedding = embedding_text(full_text)
        results = client.search(
            collection_name=colletion_name,
            query_vector={
                "text": query_embedding.tolist()
            },
            limit=top_k,
            filter = None
        )
        return [(result.payload["file_name"], result.score) for result in results]
    except Exception as e:
        print(f"Error searching image by text: {e}")
        return None
    
def create_points(image_dir , animal):
    try:
        image_dir = f"{image_dir}"
        image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        points = []
        for i, image_file in enumerate(image_files):
            image_path = os.path.join(image_dir, image_file)
            embedding_image = embedding_image(image_path)
            embedding_text = embedding_text(f"{animal.category} {animal.type_of_category} {animal.description}")
            points.append(
                get_points(
                    id=i,
                    embedding_image=embedding_image,
                    embedding_text=embedding_text,
                    image_path=image_path,
                    file_name=image_file,
                    category=f"{animal.category}",
                    description=f"{animal.description}",
                    type_of_category=f"{animal.type_of_category}"
                )
            )
        return points
    except Exception as e:
        print(f"Error creating datasets: {e}")
        return None

def create_images(points):
    try:
        for point in points:
            save_image_to_qdrant(
                collection_name="advanced_image_search",
                points=point,
                image_path=point.payload["path"]
            )
    except Exception as e:
        print(f"Error creating images: {e}")
        return None

def create_class_animal_data():
    animals = []
    try:
        images_len , files_name = count_images('images/cats')
        desc = cat_description()


        for i in range(images_len):
            animal = Animal(
                name=files_name[i],
                category="animal",
                des=desc[i],
                type="cat"
            )
            animals.append(animal)
            print(f"Animal {i+1}: {animal.name}, Category: {animal.category}, Description: {animal.description}, Type: {animal.type_of_category}")
        return animals  
    except Exception as e:
        print(f'error')
        return None    

def get_image_file_name(image_dir):
    try:
        image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        image_files.sort()  # Sort filenames alphabetically
        return image_files
    except Exception as e:
        print(f"Error getting image file names: {e}")
        return None

def create_upsert():
    # create collection
    try:
        create_collection("advanced_image_search")
    except Exception as e:
        print(f"Error creating collection: {e}")
        return None
    
    # we have all datas ( animals )
    animals = create_class_animal_data()
    if animals is None:
        print("No animals data available.")
        return None
    
    # create points
    points = []
    file_names = get_image_file_name("images/cats")
    for i , animal in animals:
        points.append(get_points(
            id = i,
            embedding_image=embedding_image(f"images/cats/{animal.name}"),
            embedding_text=embedding_text(f"{animal.category} {animal.type_of_category} {animal.description}"),
            image_path=f"images/cats/{animal.name}",
            file_name=file_names[i  - 1],
            category=animal.category,
            description=animal.description,
            type_of_category=animal.type_of_category    
        ))

    # save points to qdrant
    for point in points:
        save_image_to_qdrant(
            collection_name="advanced_image_search",
            points=point,
            image_path=point.payload["path"]
        )

    

    
def main():
    create_upsert()

    # search time
    user_query = "show me the white cat"
    result = search_image_by_query_user(
        user_query=user_query,
        collection_name="advanced_image_search",
        top_k=5
    )
    if result is not None:
        print(f"Search results for query '{user_query}':")
        for file_name, score in result:
            print(f"File: {file_name}, Score: {score}")
    else:
        print("No results found.")

    
    # search by payload
    category = "animal"
    description = "white cat"
    type_of_category = "cat"
    result = search_image_by_text_payload(
        category=category,
        description=description,
        type_of_category=type_of_category,
        colletion_name="advanced_image_search",
        query_text=f"{category} {description} {type_of_category}",
        top_k=5
    )
    if result is not None:
        print(f"Search results for payload '{category} {description} {type_of_category}':")
        for file_name, score in result:
            print(f"File: {file_name}, Score: {score}")
    else:
        print("No results found.")



if __name__ == "__main__":
    main()




    
    



"""
    create 
    is_exist 
    create_points
    get_embedding_image
    get_embedding_text
    save_image_to_qdrant 
    search_image_by_text


    now we can create our animals class and fill it and then we can create our points
    base on Animal class with correct description
"""