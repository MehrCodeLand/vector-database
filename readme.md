# Qdrant Vector Database Examples & Projects

This repository contains a collection of code examples and projects demonstrating the use of Qdrant, a powerful vector database for similarity search and machine learning applications.

## Overview

Qdrant is a vector similarity search engine that provides a production-ready service with a convenient API to store, search, and manage points—vectors with an optional payload. This repository demonstrates various ways to use Qdrant for different use cases.

## Prerequisites

- Python 3.7+
- Qdrant server running (default: localhost:6333)
- Required Python packages:
  - qdrant-client
  - sentence-transformers
  - transformers
  - torch
  - PIL (Pillow)
  - pandas
  - numpy

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/qdrant-samples.git
cd qdrant-samples

# Install dependencies
pip install -r requirements.txt

# Start Qdrant (if not already running)
# Using Docker:
docker run -p 6333:6333 qdrant/qdrant
```

## Project Structure

```
qdrant-samples/
├── simple code/
│   └── qdrant_how_to_code.py  # Basic Qdrant operations
├── project/
│   ├── semantic_search.py     # Semantic search for text documents
│   ├── image_search.py        # Image similarity search
│   └── multilingual.py        # Multilingual text search
└── README.md
```

## Projects

### 1. Basic Qdrant Operations (`simple code/qdrant_how_to_code.py`)

A simple introduction to Qdrant that covers:
- Creating a collection
- Inserting vectors with payloads
- Performing vector similarity search
- Displaying search results

This is an excellent starting point to understand the fundamental operations of Qdrant.

### 2. Semantic Search (`project/semantic_search.py`)

Demonstrates semantic search capabilities using:
- Sentence transformer models for text embeddings
- Document indexing with metadata
- Similarity search based on meaning rather than keywords
- Example with a collection of science fiction books

### 3. Image Search (`project/image_search.py`)

Implements image similarity search using:
- Vision Transformer (ViT) models for image embedding
- Image preprocessing and feature extraction
- Vector similarity for finding similar images
- Practical example with local image collection

### 4. Multilingual Search (`project/multilingual.py`)

Showcases cross-language semantic search:
- Multilingual embedding model supporting 50+ languages
- Cross-lingual similarity matching
- Filtering by language and topic
- Examples of querying in one language and finding relevant content in others

## Usage Examples

### Basic Vector Operations

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Connect to Qdrant
client = QdrantClient("localhost", port=6333)

# Create a collection
client.create_collection(
    collection_name="my_collection",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

# Insert vectors
client.upsert(
    collection_name="my_collection",
    points=[
        PointStruct(
            id=1,
            vector=[0.1, 0.2, ...],  # 384-dimensional vector
            payload={"metadata": "example"}
        )
    ]
)

# Search for similar vectors
results = client.search(
    collection_name="my_collection",
    query_vector=[0.1, 0.2, ...],  # Query vector
    limit=5
)
```

### Semantic Text Search

```python
# See project/semantic_search.py for complete implementation
from sentence_transformers import SentenceTransformer

# Initialize the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings for your text
text_embedding = model.encode("Your search query").tolist()

# Search in Qdrant
results = client.search(
    collection_name="semantic_search",
    query_vector=text_embedding,
    limit=3
)
```

### Image Similarity Search

```python
# See project/image_search.py for complete implementation
from PIL import Image
from transformers import AutoFeatureExtractor, AutoModel

# Get embedding for query image
query_embedding = get_image_embedding("path/to/query_image.jpg")

# Search for similar images
results = client.search(
    collection_name="images",
    query_vector=query_embedding.tolist(),
    limit=5
)
```

### Multilingual Search

```python
# See project/multilingual.py for complete implementation
# Search for "artificial intelligence" in any language
results = search_documents("artificial intelligence")

# Search in specific language or topic
results = search_documents("énergie renouvelable", language="en", topic="energy")
```

## Key Features

- **Vector Similarity Search**: Find semantically similar items using cosine distance
- **Cross-Modal Applications**: Text-to-text, image-to-image similarity
- **Multilingual Support**: Search across language barriers
- **Metadata Filtering**: Combine vector search with metadata constraints
- **Efficient Indexing**: Fast retrieval even with large vector collections

## Applications

This repository demonstrates implementation patterns for:
- Semantic document search
- Similar image retrieval
- Multilingual information retrieval
- Product recommendation systems
- Content-based filtering

## Resources

- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Sentence Transformers](https://www.sbert.net/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
