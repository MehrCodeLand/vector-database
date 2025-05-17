from qdrant_client import QdrantClient
from qdrant_client.models import Distance , VectorParams

client = QdrantClient("localhost", port=6333)


from qdrant_client.models import PointStruct


def create_collection():
    # create a collection
    client.create_collection(
        collection_name="test_collection_1",
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE,
        )
    )

def create_products():
    products = [
        {
            "id": 1,
            "vector": [0.1, 0.2] + [0.0] * 380 + [0.3, 0.4],
            "payload": {
                "name": "Blue T-shirt",
                "category": "clothing",
                "price": 25.99
            }
        },
        {
            "id": 2,
            "vector": [0.5, 0.6] + [0.0] * 380 + [0.7, 0.8],
            "payload": {
                "name": "Red Sneakers",
                "category": "footwear",
                "price": 49.99
            }
        },
        {
            "id": 3,
            "vector": [0.9, 1.0] + [0.0] * 380 + [1.1, 1.2],
            "payload": {
                "name": "Green Hat",
                "category": "accessories",
                "price": 15.99
            }
        },
    ]

    return products

def upsert_vectors(products):
    client.upsert(
    collection_name="test_collection_1",
    points=[
        PointStruct(
            id=products["id"],
            vector=products["vector"],
            payload=products["payload"],
        )
        for products in products
    ]
)

def run():
    create_collection()
    products = create_products()
    upsert_vectors(products)



def search():
    query_vector = [0.9, 1.0] + [0.0] * 380 + [1.1, 1.2]
    search_result = client.search(
        collection_name = "test_collection_1",
        query_vector=query_vector,
        limit=3,
    )
    return search_result

def show_result(result):
    for hit in result:
        print(f"ID: {hit.id}, Score: {hit.score}, Payload: {hit.payload}")


res = search()
show_result(res)