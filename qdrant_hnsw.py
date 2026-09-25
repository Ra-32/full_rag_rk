import os
import warnings
import json

from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    PayloadSchemaType
)


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

warnings.filterwarnings(
    "ignore",
    message="IProgress not found"
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing from .env")

if not QDRANT_URL:
    raise ValueError("QDRANT_URL is missing from .env")

if not QDRANT_API_KEY:
    raise ValueError("QDRANT_API_KEY is missing from .env")


print("API credentials loaded successfully")


# ============================================================
# CONNECT TO QDRANT
# ============================================================

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

print("Connected to Qdrant Cloud")


# ============================================================
# COLLECTION CONFIGURATION
# ============================================================

COLLECTION_NAME = "knowledge_filter"
EMBEDDING_SIZE = 384


# ============================================================
# DELETE EXISTING COLLECTION
# ============================================================

if qdrant_client.collection_exists(COLLECTION_NAME):

    print(
        f"Deleting existing collection: "
        f"{COLLECTION_NAME}"
    )

    qdrant_client.delete_collection(
        COLLECTION_NAME
    )


# ============================================================
# CREATE COLLECTION
# ============================================================

qdrant_client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=EMBEDDING_SIZE,
        distance=Distance.COSINE
    )
)

print(f"Created collection: {COLLECTION_NAME}")
print(f"Vector size: {EMBEDDING_SIZE}")
print("Distance: COSINE")


# ============================================================
# CREATE PAYLOAD INDEXES
# ============================================================

qdrant_client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="category",
    field_schema=PayloadSchemaType.KEYWORD
)

qdrant_client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="is_active",
    field_schema=PayloadSchemaType.BOOL
)

print("Payload indexes created")


# ============================================================
# LOAD KNOWLEDGE BASE
# ============================================================

with open(
    "knowledge.json",
    "r",
    encoding="utf-8"
) as f:

    documents = json.load(f)


print(
    f"Loaded {len(documents)} documents"
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model is ready")


# ============================================================
# CREATE DOCUMENT TEXT LIST
# ============================================================

texts = [
    document["text"]
    for document in documents
]


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

embeddings = embedding_model.encode(texts)

print(
    f"Generated {len(embeddings)} embeddings"
)

print(
    f"Embedding size: {len(embeddings[0])}"
)


# ============================================================
# CREATE QDRANT POINTS
# ============================================================

points = []

for i in range(len(documents)):

    point = PointStruct(
        id=i + 1,

        # IMPORTANT:
        # Use only the embedding for this document
        vector=embeddings[i].tolist(),

        # Store the complete JSON object
        # as Qdrant payload
        payload=documents[i]
    )

    points.append(point)


# ============================================================
# UPLOAD POINTS TO QDRANT
# ============================================================

qdrant_client.upsert(
    collection_name=COLLECTION_NAME,
    points=points
)

print(
    f"Uploaded {len(points)} documents to Qdrant"
)


# ============================================================
# NORMAL SEARCH
# ============================================================

def search(query, top_k=3):

    query_vector = embedding_model.encode(
        query
    ).tolist()

    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True
    ).points

    return results


# ============================================================
# FILTERED SEARCH
# ============================================================

def search_with_filter(
    query,
    query_filter=None,
    top_k=3
):

    query_vector = embedding_model.encode(
        query
    ).tolist()

    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
        query_filter=query_filter
    ).points

    return results


# ============================================================
# LEAVE + ACTIVE FILTER
# ============================================================

leave_filter = Filter(
    must=[

        # category = leave
        FieldCondition(
            key="category",
            match=MatchValue(
                value="leave"
            )
        ),

        # is_active = true
        FieldCondition(
            key="is_active",
            match=MatchValue(
                value=True
            )
        )
    ]
)


# ============================================================
# TEST FILTERED SEARCH
# ============================================================

query = (
    "How many paid leave days do "
    "employees receive per year?"
)

results = search_with_filter(
    query,
    leave_filter,
    top_k=3
)


print("\n")
print("=" * 60)
print("FILTERED SEARCH RESULTS")
print("=" * 60)


for result in results:

    print(
        f"Score: {result.score:.3f}"
    )

    print(
        f"Category: "
        f"{result.payload['category']}"
    )

    print(
        f"Active: "
        f"{result.payload['is_active']}"
    )

    print(
        f"Text: "
        f"{result.payload['text']}"
    )

    print()


# ============================================================
# GROQ CLIENT
# ============================================================

groq_client = Groq(
    api_key=GROQ_API_KEY
)


# ============================================================
# ASK LLM
# ============================================================

def ask_llm(question, context):

    prompt = f"""
You are a helpful question-answering assistant.

Answer the user's question using ONLY the information
provided in the context.

If the answer is not present in the context, say:

"I don't know based on the provided information."

Do not use outside knowledge.

Context:
{context}

Question:
{question}

Answer:
"""

    response = groq_client.chat.completions.create(

        model="openai/gpt-oss-120b",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0
    )

    return response.choices[0].message.content


# ============================================================
# COMPLETE RAG PIPELINE
# ============================================================

question = (
    "How far in advance should I request leave?"
)


# Search only ACTIVE leave policies
results = search_with_filter(
    question,
    leave_filter,
    top_k=3
)


# Create context
context = "\n".join(
    result.payload["text"]
    for result in results
)


# Ask LLM
answer = ask_llm(
    question,
    context
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 60)
print("FINAL ANSWER")
print("=" * 60)

print(answer)