# Qdrant: Filtering and HNSW

## Overview

Qdrant is a vector database designed for storing, indexing, and searching high-dimensional vectors.

In a RAG application, documents are converted into embeddings and stored in a Qdrant collection. When a user asks a question, the question is also converted into an embedding, and Qdrant searches for the most similar stored vectors.

Two important concepts when working with Qdrant are:

1. **HNSW** — Efficient approximate nearest-neighbor vector search
2. **Payload Filtering** — Restricting vector search to documents that satisfy specific conditions

---

# 1. HNSW

## What is HNSW?

**HNSW** stands for:

> **Hierarchical Navigable Small World**

HNSW is a graph-based algorithm used for **approximate nearest-neighbor (ANN) search**.

The main purpose of HNSW is to find vectors that are close to a query vector without having to compare the query against every vector in the collection.

---

## Why Do We Need HNSW?

Imagine that a Qdrant collection contains:

```text
1,000,000 vectors
```

A simple brute-force search could compare the query vector against:

```text
Vector 1
Vector 2
Vector 3
Vector 4
...
Vector 1,000,000
```

This is called a **linear / exhaustive search** approach.

As the number of vectors increases, checking every vector becomes expensive.

HNSW creates a graph structure that allows the search to navigate toward promising vectors instead of examining every vector.

---

# 2. Real-World Example

Imagine you are in **Bengaluru** and want to find a particular **Kalyan Jewellers in Delhi**.

### Exhaustive / Linear Search

You could imagine checking every possible location:

```text
Street 1
   ↓
Street 2
   ↓
Street 3
   ↓
Street 4
   ↓
Street 5
   ↓
...
   ↓
Kalyan Jewellers
```

The idea is that many locations may be checked before reaching the destination.

---

## HNSW-style Navigation

Instead, imagine having a navigation system that first identifies larger regions and then progressively moves toward the destination:

```text
Bengaluru
    │
    ▼
Delhi
    │
    ▼
Central Delhi
    │
    ▼
Nearby Area
    │
    ▼
Kalyan Jewellers
```

The important idea is:

> **Navigate through increasingly relevant/closer candidates rather than examining every possible location.**

This is analogous to how HNSW navigates through its graph.

---

# 3. HNSW and Vector Search

Suppose we have document embeddings:

```text
Document A → Vector
Document B → Vector
Document C → Vector
Document D → Vector
Document E → Vector
...
```

The user asks:

```text
"What is the company's leave policy?"
```

The question is converted into a query vector:

```text
Query → [0.21, -0.14, 0.37, ...]
```

Qdrant needs to find vectors close to this query vector.

The distance metric determines **how closeness is measured**.

For example:

```text
Cosine similarity
```

HNSW determines **how the search navigates efficiently through the vector index**.

Therefore:

```text
Distance Metric
      +
HNSW Index
      ↓
Efficient Vector Search
```

---

# 4. HNSW Does Not Replace Cosine Similarity

This is an important distinction.

It is incorrect to think:

```text
HNSW instead of cosine similarity
```

A better understanding is:

```text
                 Vector Search
                      │
          ┌───────────┴───────────┐
          │                       │
     HNSW Index             Distance Metric
          │                       │
   Search efficiently       Measure similarity
          │                       │
          └───────────┬───────────┘
                      ▼
               Search Results
```

For example, your Qdrant collection uses:

```python
VectorParams(
    size=384,
    distance=Distance.COSINE
)
```

Here:

* `384` → vector dimensionality
* `COSINE` → similarity/distance metric
* HNSW → indexing/search structure used for efficient approximate nearest-neighbor search

---

# 5. Why HNSW Is Useful in RAG

Consider a RAG system with:

```text
10 documents
```

A brute-force search may be perfectly acceptable.

But imagine:

```text
1 million documents
```

or:

```text
100 million vectors
```

Searching every vector becomes increasingly expensive.

HNSW provides a graph-based structure that can make nearest-neighbor search much more efficient.

This makes it useful for:

* RAG systems
* Semantic search
* Recommendation systems
* Image similarity search
* Document retrieval
* Large-scale vector databases

---

# 6. Qdrant Collections

A Qdrant collection is conceptually similar to a table in a traditional database.

For example:

```text
Qdrant
│
└── Collection
      │
      ├── Point 1
      ├── Point 2
      ├── Point 3
      └── Point 4
```

Each point can contain:

```text
Point
├── ID
├── Vector
└── Payload
```

For your company-policy RAG:

```json
{
  "text": "Employees receive 24 days of paid leave per year.",
  "category": "leave",
  "is_active": true
}
```

The payload stores metadata associated with the vector.

---

# 7. Qdrant Payload Filtering

Vector similarity alone is not always enough.

Suppose your company knowledge base contains:

```text
HR policies
IT policies
Finance policies
Leave policies
Security policies
```

Each document can have metadata:

```text
text
category
is_active
```

For example:

```json
{
  "text": "Employees receive 24 days of paid leave per year.",
  "category": "leave",
  "is_active": true
}
```

You can use this metadata to restrict the search.

---

# 8. Why Use Filters?

Suppose the user asks:

```text
How many paid leave days do employees receive?
```

Instead of searching through every company policy, we can tell Qdrant:

```text
Only search documents where:

category = "leave"
```

We can also add:

```text
is_active = true
```

So the search becomes:

```text
User Query
    │
    ▼
Query Embedding
    │
    ▼
Qdrant
    │
    ├── category = "leave"
    │
    └── is_active = true
    │
    ▼
Vector Similarity Search
    │
    ▼
Top-K Results
```

---

# 9. Creating Payload Indexes

If you frequently filter on a payload field, create a payload index.

For example:

```python
qdrant_client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="category",
    field_schema=PayloadSchemaType.KEYWORD
)
```

For the `is_active` field:

```python
qdrant_client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="is_active",
    field_schema=PayloadSchemaType.BOOL
)
```

Now Qdrant has indexes for these fields.

```text
Collection
│
├── Vector Index
│     └── HNSW
│
├── Payload Index
│     ├── category
│     └── is_active
│
└── Points
```

---

# 10. Qdrant Filter Conditions

Qdrant provides logical conditions for combining filters.

The three important concepts are:

```text
must
must_not
should
```

---

# 11. `must`

`must` means that **all specified conditions must be satisfied**.

Think of it as:

```text
Condition A
    AND
Condition B
```

### Example

We want:

```text
category = "leave"
AND
is_active = true
```

Python:

```python
leave_filter = Filter(
    must=[
        FieldCondition(
            key="category",
            match=MatchValue(
                value="leave"
            )
        ),

        FieldCondition(
            key="is_active",
            match=MatchValue(
                value=True
            )
        )
    ]
)
```

Conceptually:

```text
             Document
                │
        ┌───────┴────────┐
        │                │
 category = leave?   is_active = true?
        │                │
       YES              YES
        │                │
        └───────┬────────┘
                ▼
             INCLUDE
```

If either condition is false, the document does not satisfy the `must` filter.

---

# 12. `must_not`

`must_not` means that matching documents should be excluded.

For example, suppose we want to search all policies **except vacation policies**.

```python
vacation_exclusion_filter = Filter(
    must_not=[
        FieldCondition(
            key="category",
            match=MatchValue(
                value="vacations"
            )
        )
    ]
)
```

Conceptually:

```text
category = vacations?
        │
        ├── YES → EXCLUDE
        │
        └── NO  → ALLOW
```

This is useful when you want to exclude certain categories or metadata values.

---

# 13. `should`

`should` is used when multiple alternative conditions are acceptable.

Conceptually:

```text
Condition A
     OR
Condition B
```

For example, suppose we want documents belonging to either:

```text
leave
```

or:

```text
holidays
```

A `should` condition can express this kind of alternative matching.

Conceptually:

```text
             Document
                │
        ┌───────┴────────┐
        │                │
 category = leave?   category = holidays?
        │                │
       YES              YES
        │                │
        └───────┬────────┘
                ▼
              MATCH
```

The exact behavior of `should` can depend on how it is combined with other filter clauses, so it is useful to think of it as expressing **alternative conditions**, rather than simply assuming it always means a standalone SQL-style `OR`.

---

# 14. Filter Example for HR Policies

Suppose the knowledge base contains:

```text
Document 1
category = leave
is_active = true

Document 2
category = leave
is_active = false

Document 3
category = salary
is_active = true

Document 4
category = security
is_active = true
```

We want:

```text
category = leave
AND
is_active = true
```

The result should contain:

```text
Document 1
```

and exclude:

```text
Document 2 → inactive
Document 3 → wrong category
Document 4 → wrong category
```

---

# 15. Filtering + HNSW Together

The complete Qdrant search process can be understood as:

```text
                 User Question
                       │
                       ▼
                Embedding Model
                       │
                       ▼
                  Query Vector
                       │
                       ▼
              ┌─────────────────┐
              │     Qdrant      │
              │                 │
              │    Filters      │
              │ category=leave  │
              │ is_active=true  │
              │                 │
              │      ↓          │
              │     HNSW        │
              │      ↓          │
              │ Similarity      │
              │ Search          │
              └────────┬────────┘
                       │
                       ▼
                  Top-K Results
                       │
                       ▼
                    Context
                       │
                       ▼
                     LLM
                       │
                       ▼
                  Final Answer
```

The important idea is:

> **Filters restrict which points are eligible, while vector search finds the most relevant vectors among the eligible points.**

---

# 16. Complete Filtered Search Example

```python
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue
)

leave_filter = Filter(
    must=[
        FieldCondition(
            key="category",
            match=MatchValue(
                value="leave"
            )
        ),
        FieldCondition(
            key="is_active",
            match=MatchValue(
                value=True
            )
        )
    ]
)
```

Then:

```python
results = qdrant_client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    query_filter=leave_filter,
    limit=3,
    with_payload=True
).points
```

The search now means:

```text
Find the 3 most relevant vectors

BUT ONLY FROM documents where:

category = leave
AND
is_active = true
```

---

# 17. Filter Types — Quick Summary

| Filter     | Meaning                          | Example             |
| ---------- | -------------------------------- | ------------------- |
| `must`     | All conditions must match        | `leave AND active`  |
| `must_not` | Matching conditions are excluded | `NOT vacation`      |
| `should`   | Alternative conditions can match | `leave OR holidays` |

---

# 18. HNSW vs Filtering

These two concepts solve different problems.

| Feature           | Purpose                                  |
| ----------------- | ---------------------------------------- |
| HNSW              | Efficient vector nearest-neighbor search |
| Cosine similarity | Measures vector similarity               |
| Payload           | Stores metadata                          |
| Payload index     | Makes metadata filtering efficient       |
| `must`            | Requires conditions                      |
| `must_not`        | Excludes conditions                      |
| `should`          | Allows alternative conditions            |

Therefore:

```text
HNSW ≠ Filter
```

and:

```text
HNSW ≠ Cosine Similarity
```

Instead:

```text
HNSW
   ↓
Efficiently navigate vector search

Cosine
   ↓
Measure vector similarity

Filter
   ↓
Restrict eligible documents
```

---

# 19. RAG Example

For a company-policy RAG system:

### Knowledge Base

```json
{
  "text": "Employees receive 24 days of paid leave per year.",
  "category": "leave",
  "is_active": true
}
```

### User Question

```text
How many paid leave days do employees receive?
```

### Processing

```text
Question
   ↓
Embedding
   ↓
Qdrant
   ↓
Filter:
category = leave
is_active = true
   ↓
HNSW Vector Search
   ↓
Top-K Relevant Documents
   ↓
Context
   ↓
LLM
   ↓
Answer
```

### Final Answer

```text
Employees receive 24 days of paid leave per year.
```

---

# 20. Key Takeaways

### HNSW

```text
HNSW = Hierarchical Navigable Small World
```

It is a graph-based approximate nearest-neighbor search algorithm/index structure that helps Qdrant efficiently search large vector collections.

### Distance Metric

```text
Cosine
```

measures how similar vectors are.

### Payload

```text
text
category
is_active
```

stores metadata associated with each vector.

### Payload Index

```text
category → KEYWORD
is_active → BOOL
```

allows efficient filtering on metadata fields.

### Filters

```text
must
must_not
should
```

allow you to control which documents are eligible for retrieval.

### Complete Mental Model

```text
                 QDRANT
                   │
       ┌───────────┴───────────┐
       │                       │
   PAYLOAD                  VECTOR
   FILTERING                SEARCH
       │                       │
       │                     HNSW
       │                       │
       │                 Cosine Distance
       │                       │
       └───────────┬───────────┘
                   ▼
             Top-K Results
                   │
                   ▼
                 RAG
                   │
                   ▼
                  LLM
                   │
                   ▼
                Answer
```

> **In a RAG system, Qdrant can combine metadata filtering with vector similarity search so that the system retrieves semantically relevant documents from the appropriate subset of the knowledge base.**
