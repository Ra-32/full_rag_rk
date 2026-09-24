import os
import warnings

from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

warnings.filterwarnings("ignore", message="IProgress not found")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


# Check environment variables
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing from .env")

if not QDRANT_URL:
    raise ValueError("QDRANT_URL is missing from .env")

if not QDRANT_API_KEY:
    raise ValueError("QDRANT_API_KEY is missing from .env")

print("API credentials loaded successfully")


# ============================================================
# 2. CONNECT TO QDRANT CLOUD
# ============================================================

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

print("Connected to Qdrant Cloud")


# ============================================================
# 3. CONFIGURATION
# ============================================================

COLLECTION_NAME = "knowledge"
EMBEDDING_SIZE = 384

KNOWLEDGE_FILE = "knowledege.txt"


# ============================================================
# 4. CREATE / RESET QDRANT COLLECTION
# ============================================================

if qdrant_client.collection_exists(COLLECTION_NAME):
    print(f"Deleting existing collection: {COLLECTION_NAME}")
    qdrant_client.delete_collection(COLLECTION_NAME)


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
# 5. LOAD KNOWLEDGE BASE
# ============================================================

with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as file:

    documents = [
        line.strip()
        for line in file
        if line.strip()
    ]


print(f"Loaded {len(documents)} documents")


# ============================================================
# 6. LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model is ready")


# ============================================================
# 7. CREATE EMBEDDINGS
# ============================================================

embeddings = embedding_model.encode(documents)

print(f"Generated {len(embeddings)} embeddings")
print(f"Embedding size: {len(embeddings[0])}")


# ============================================================
# 8. CREATE QDRANT POINTS
# ============================================================

points = []

for index, embedding in enumerate(embeddings):

    point = PointStruct(
        id=index + 1,
        vector=embedding.tolist(),
        payload={
            "text": documents[index]
        }
    )

    points.append(point)


# ============================================================
# 9. UPLOAD VECTORS TO QDRANT
# ============================================================

qdrant_client.upsert(
    collection_name=COLLECTION_NAME,
    points=points
)

print(f"Uploaded {len(points)} documents to Qdrant")


# ============================================================
# 10. SEARCH FUNCTION
# ============================================================

def search(query, top_k=3):

    # Convert question into embedding
    query_vector = embedding_model.encode(query).tolist()

    # Search Qdrant
    search_results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True
    ).points

    return search_results


# ============================================================
# 11. CONNECT TO GROQ
# ============================================================

groq_client = Groq(
    api_key=GROQ_API_KEY
)

print("Connected to Groq")


# ============================================================
# 12. LLM FUNCTION
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
# 13. COMPLETE RAG PIPELINE
# ============================================================

question = "What is RAG?"

print("\n" + "=" * 60)
print("QUESTION")
print("=" * 60)

print(question)


# Retrieve relevant documents
results = search(
    question,
    top_k=3
)


# ============================================================
# 14. DISPLAY RETRIEVED DOCUMENTS
# ============================================================

print("\n" + "=" * 60)
print("RETRIEVED DOCUMENTS")
print("=" * 60)

for index, result in enumerate(results, start=1):

    print(f"\nDocument {index}")
    print(f"Score: {result.score:.4f}")
    print(f"Text: {result.payload['text']}")


# ============================================================
# 15. CREATE CONTEXT
# ============================================================

context = "\n".join(
    result.payload["text"]
    for result in results
)


print("\n" + "=" * 60)
print("CONTEXT SENT TO LLM")
print("=" * 60)

print(context)


# ============================================================
# 16. GENERATE FINAL ANSWER
# ============================================================

answer = ask_llm(
    question,
    context
)


print("\n" + "=" * 60)
print("FINAL ANSWER")
print("=" * 60)

print(answer)