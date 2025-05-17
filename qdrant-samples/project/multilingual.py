from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

import pandas as pd
import uuid


model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')  # Supports 50+ languages
client = QdrantClient("localhost", port=6333)


vector_size = model.get_sentence_embedding_dimension()
client.create_collection(
    collection_name="multilingual_docs",
    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
)



client.create_payload_index(
    collection_name="multilingual_docs",
    field_name="language",
    field_schema="keyword"
)
client.create_payload_index(
    collection_name="multilingual_docs",
    field_name="topic",
    field_schema="keyword"
)



documents = [
    {"text": "Machine learning is a subset of artificial intelligence.", "language": "en", "topic": "technology"},
    {"text": "El aprendizaje automático es un subconjunto de la inteligencia artificial.", "language": "es", "topic": "technology"},
    {"text": "机器学习是人工智能的一个子集。", "language": "zh", "topic": "technology"},
    {"text": "Climate change is affecting global weather patterns.", "language": "en", "topic": "environment"},
    {"text": "Le changement climatique affecte les modèles météorologiques mondiaux.", "language": "fr", "topic": "environment"},
    {"text": "太陽エネルギーは再生可能エネルギーの一種です。", "language": "ja", "topic": "energy"},
    {"text": "Solar energy is a type of renewable energy.", "language": "en", "topic": "energy"},
]


embeddings = model.encode([doc["text"] for doc in documents])
points = [
    PointStruct(
        id=str(uuid.uuid4()),
        vector=embedding.tolist(),
        payload={
            "text": doc["text"],
            "language": doc["language"],
            "topic": doc["topic"]
        }
    )
    for doc, embedding in zip(documents, embeddings)
]
client.upsert(collection_name="multilingual_docs", points=points)

def search_documents(query, language=None, topic=None, top_k=3):
    query_vector = model.encode(query).tolist()
    
    # Build list of FieldCondition
    conditions = []
    if language:
        conditions.append(
            FieldCondition(key="language", match=MatchValue(value=language))
        )
    if topic:
        conditions.append(
            FieldCondition(key="topic", match=MatchValue(value=topic))
        )
    
    # Wrap in a Filter if we have any
    query_filter = Filter(must=conditions) if conditions else None
    
    # Pass query_filter (not filter) to client.search
    results = client.search(
        collection_name="multilingual_docs",
        query_vector=query_vector,
        limit=top_k,
        query_filter=query_filter
    )
    
    return [
        {
            "text":    hit.payload["text"],
            "language": hit.payload.get("language"),
            "topic":    hit.payload.get("topic"),
            "score":    hit.score
        }
        for hit in results
    ]

    

print("Search for 'artificial intelligence' in any language:")
results = search_documents("artificial intelligence")
for r in results:
    print(f"[{r['language']}] {r['text']} - Score: {r['score']:.4f}")

print("\nSearch for 'énergie renouvelable' restricted to English:")
results = search_documents("énergie renouvelable", language="en")
for r in results:
    print(f"[{r['language']}] {r['text']} - Score: {r['score']:.4f}")

print("\nSearch for '气候' in the environment topic:")
results = search_documents("气候", topic="environment")
for r in results:
    print(f"[{r['language']}] {r['text']} - Score: {r['score']:.4f}")