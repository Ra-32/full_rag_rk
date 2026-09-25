# Qdrant RAG Pipeline

A simple **Retrieval-Augmented Generation (RAG)** pipeline using:

* **Qdrant** — Vector database for storing and searching embeddings
* **Sentence Transformers** — Converts documents and queries into embeddings
* **Groq** — LLM inference
* **OpenAI GPT-OSS 120B** — Used as the generation model
* **Python** — Pipeline implementation

The pipeline follows the standard RAG architecture:

**Knowledge Base → Embeddings → Qdrant → Similarity Search → Context → LLM → Answer**

---

## 📌 Architecture

```text
                Knowledge Base
                      │
                      ▼
              Read Documents
                      │
                      ▼
          SentenceTransformer
                      │
                      ▼
                Embeddings
                      │
                      ▼
        ┌─────────────────────────┐
        │         Qdrant          │
        │                         │
        │  Collection             │
        │    ├── Point ID         │
        │    ├── Vector            │
        │    └── Payload           │
        └─────────────────────────┘
                      │
                User Question
                      │
                      ▼
          Convert Question to
               Embedding
                      │
                      ▼
             Similarity Search
                      │
                      ▼
             Top-K Documents
                      │
                      ▼
                   Context
                      │
                      ▼
                 Groq LLM
                      │
                      ▼
                Final Answer
```

---

# 🔄 RAG Pipeline Steps

## 1. Initialize Qdrant Client

First, connect to the Qdrant server using the Qdrant URL and API key.

```python
from qdrant_client import QdrantClient

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)
```

---

## 2. Check and Recreate the Collection

Before inserting data, check whether the collection already exists.

If it exists, delete it and create a fresh collection.

```python
if qdrant_client.collection_exists(COLLECTION_NAME):
    qdrant_client.delete_collection(COLLECTION_NAME)

qdrant_client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=EMBEDDING_SIZE,
        distance=Distance.COSINE
    )
)
```

### Why?

This ensures that the collection contains only the current knowledge-base data.

### Vector Configuration

```text
size = 384
distance = COSINE
```

For example, when using:

```python
SentenceTransformer("all-MiniLM-L6-v2")
```

the generated embedding size is **384 dimensions**.

---

# 3. Load the Knowledge Base

The knowledge base is stored in a text file.

Each non-empty line is treated as one document.

```python
with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as file:

    documents = [
        line.strip()
        for line in file
        if line.strip()
    ]
```

For example:

```text
Python is a high-level programming language.
RAG stands for Retrieval-Augmented Generation.
Qdrant is a vector database.
Embeddings represent text as numerical vectors.
```

Each line becomes an individual document.

---

# 4. Load the Embedding Model

Use Sentence Transformers to convert text into numerical vectors.

```python
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)
```

The model converts text such as:

```text
What is RAG?
```

into a numerical vector:

```text
[0.021, -0.134, 0.452, ...]
```

with **384 dimensions**.

---

# 5. Generate Document Embeddings

Convert all documents into embeddings.

```python
embeddings = embedding_model.encode(documents)
```

If there are 15 documents:

```text
15 documents
      ↓
15 embeddings
      ↓
Each embedding = 384 dimensions
```

---

# 6. SQL Tables vs Qdrant Collections

A useful way to understand Qdrant is to compare it with a traditional SQL database.

| SQL Database        | Qdrant                   |
| ------------------- | ------------------------ |
| Database            | Qdrant instance          |
| Table               | Collection               |
| Row                 | Point                    |
| Primary Key         | Point ID                 |
| Columns             | Payload                  |
| Numeric/vector data | Vector                   |
| WHERE query         | Vector similarity search |

A Qdrant point contains:

```text
Point
├── ID
├── Vector
└── Payload
    └── text
```

Example:

```python
{
    "id": 1,
    "vector": [0.021, -0.134, ...],
    "payload": {
        "text": "RAG stands for Retrieval-Augmented Generation."
    }
}
```

---

# 7. Create Qdrant Points

Create a `PointStruct` for every document.

```python
from qdrant_client.models import PointStruct

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
```

Each point contains:

* `id` → Unique point identifier
* `vector` → Document embedding
* `payload` → Original document text

---

# 8. Upload Points to Qdrant

Upload all points into the collection.

```python
qdrant_client.upsert(
    collection_name=COLLECTION_NAME,
    points=points
)
```

The data is now stored inside Qdrant.

Conceptually:

```text
Qdrant Collection
│
├── Point 1
│   ├── Vector
│   └── Text
│
├── Point 2
│   ├── Vector
│   └── Text
│
├── Point 3
│   ├── Vector
│   └── Text
│
└── ...
```

---

# 9. Create the Search Function

When the user asks a question, the question must first be converted into an embedding.

```python
def search(query, top_k=3):

    # Convert question into embedding
    query_vector = embedding_model.encode(
        query
    ).tolist()

    # Search Qdrant
    search_results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True
    ).points

    return search_results
```

### What happens here?

Suppose the user asks:

```text
What is RAG?
```

The pipeline performs:

```text
"What is RAG?"
      ↓
Embedding Model
      ↓
Query Vector
      ↓
Qdrant
      ↓
Similarity Search
      ↓
Top 3 Relevant Documents
```

The `top_k` parameter controls how many documents are retrieved.

For example:

```python
search(query, top_k=3)
```

returns the three most similar documents.

---

# 10. Connect to Groq

Create the Groq client using the API key.

```python
from groq import Groq

groq_client = Groq(
    api_key=GROQ_API_KEY
)
```

The retrieved documents will be passed to the LLM as context.

---

# 11. Create the LLM Function

The `ask_llm()` function receives:

* `question`
* `context`

and asks the LLM to generate an answer using only the retrieved information.

```python
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
```

### Important

The prompt instructs the LLM to:

1. Use only retrieved context.
2. Avoid outside knowledge.
3. Say `"I don't know based on the provided information."` when the answer is not available.

---

# 12. Complete the RAG Pipeline

Now connect retrieval and generation together.

```python
results = search(
    query,
    top_k
)

context = "\n".join(
    result.payload["text"]
    for result in results
)

answer = ask_llm(
    question,
    context
)

print(answer)
```

The complete flow is:

```text
User Question
      │
      ▼
Embedding Model
      │
      ▼
Query Embedding
      │
      ▼
Qdrant Similarity Search
      │
      ▼
Top-K Documents
      │
      ▼
Retrieved Context
      │
      ▼
Groq
      │
      ▼
GPT-OSS 120B
      │
      ▼
Final Answer
```

---

# 🧠 Complete RAG Concept

The pipeline can be divided into two major stages.

## Stage 1 — Indexing

This happens when we prepare the knowledge base.

```text
Knowledge Base
      ↓
Documents
      ↓
Embedding Model
      ↓
Document Embeddings
      ↓
Qdrant
      ↓
Stored Points
```

Each Qdrant point contains:

```text
ID
Vector
Payload
```

---

## Stage 2 — Retrieval + Generation

This happens when a user asks a question.

```text
User Question
      ↓
Query Embedding
      ↓
Qdrant Search
      ↓
Top-K Relevant Documents
      ↓
Context
      ↓
LLM
      ↓
Answer
```

---

# 📊 Example

### Knowledge Base

```text
RAG stands for Retrieval-Augmented Generation.
RAG combines information retrieval with language generation.
Qdrant is a vector database.
Embeddings convert text into numerical representations.
```

### User Question

```text
What is RAG?
```

### Retrieval

Qdrant searches for vectors similar to the question.

```text
Document 1
Score: 0.49
RAG stands for Retrieval-Augmented Generation.

Document 2
Score: 0.46
RAG combines information retrieval with language generation.

Document 3
Score: 0.42
Embeddings convert text into numerical representations.
```

### Context

```text
RAG stands for Retrieval-Augmented Generation.
RAG combines information retrieval with language generation.
Embeddings convert text into numerical representations.
```

### LLM

The retrieved context is provided to the LLM.

### Final Answer

```text
RAG stands for Retrieval-Augmented Generation.
It combines information retrieval with language generation
to provide relevant information to a language model.
```

---

# 🛠️ Technologies Used

| Technology            | Purpose                       |
| --------------------- | ----------------------------- |
| Python                | Programming language          |
| Qdrant                | Vector database               |
| Sentence Transformers | Text embeddings               |
| `all-MiniLM-L6-v2`    | Embedding model               |
| Groq                  | LLM API                       |
| GPT-OSS 120B          | Generation model              |
| Cosine Similarity     | Vector similarity measurement |

---

# 📁 Suggested Project Structure

```text
rag_ai/
│
├── knowledge.txt
├── rag.py
├── .env
├── pyproject.toml
├── README.md
└── .gitignore
```

---

# 🔐 Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
```

Never commit `.env` to GitHub.

Add it to `.gitignore`:

```gitignore
.env
.venv/
__pycache__/
.ipynb_checkpoints/
```

---

# 🚀 Key Takeaway

This project demonstrates a complete **Retrieval-Augmented Generation pipeline**:

```text
          INDEXING
             │
             ▼
       Knowledge Base
             │
             ▼
        Embeddings
             │
             ▼
          Qdrant
             │
             │
       ──────┼──────
             │
             ▼
         USER QUERY
             │
             ▼
       Query Embedding
             │
             ▼
      Qdrant Retrieval
             │
             ▼
        Top-K Context
             │
             ▼
           Groq LLM
             │
             ▼
        Final Answer
```

The key idea is:

> **Qdrant retrieves relevant information, while the LLM uses that retrieved information to generate the final answer.**
