# Chunking Strategies in RAG

**Chunking** means dividing a large document into smaller pieces called **chunks** before storing them in a vector database.

The main goal of chunking is:

> **Keep each chunk small enough for efficient retrieval, but large enough to preserve its meaning.**

If the chunk is **too small**, the meaning can be broken.

If the chunk is **too large**, the retrieved result may contain too much unrelated information.

---

## 1. Fixed-Size Chunking

In fixed-size chunking, we divide text based on a fixed number of words, characters, or tokens.

### Example

Suppose we use **50 words per chunk**:

```text
50 words → Chunk 1
50 words → Chunk 2
50 words → Chunk 3
...
```

This is a simple and strict chunking method.

### Problem

Consider:

```text
The man who did the crime was John.
```

If we split at the wrong position:

```text
Chunk 1 → The man who did the crime was

Chunk 2 → John.
```

Now, if the user asks:

```text
Who did the crime?
```

The answer **"John"** is separated from the rest of the sentence.

This can make retrieval less effective.

### Key point

```text
Too small → Meaning gets broken
Too large → Too much unrelated information
```

---

# 2. Separator-Based Chunking

Instead of using a fixed number of words, we divide the document using natural separators.

Common separators include:

```text
Paragraph
Sentence
Line break
Section
Heading
```

For example:

```text
Paragraph 1 → Chunk 1
Paragraph 2 → Chunk 2
Paragraph 3 → Chunk 3
```

### Problem

Paragraph sizes can be very different.

For example:

```text
Paragraph 1 → 20 lines
Paragraph 2 → 2000 lines
```

If we use one paragraph as one chunk, the second chunk becomes extremely large.

So separator-based chunking can still create chunks that are **too large or too small**.

---

# 3. Recursive Chunking

Recursive chunking uses **multiple separators** and tries to create chunks that stay within a maximum size.

For example, we can define separators like:

```text
1. Paragraph
2. Sentence
3. Word
4. Character
```

The algorithm first tries to split using the largest/natural separator.

If the resulting piece is still too large, it recursively splits it using the next separator.

### Simple idea

```text
Large Document
       ↓
   Paragraph
       ↓
   Sentence
       ↓
     Word
       ↓
   Character
```

For example:

```text
A B C D E F G H I J K
```

First:

```text
A B C D E F G       H I J K
```

If the first part is still too large:

```text
A B C       D E F G
```

The process continues until the chunks satisfy the required size.

### Advantage

Recursive chunking tries to preserve natural meaning while also controlling chunk size.

---

# 4. Chunk Overlap

Chunk overlap means allowing some text from the previous chunk to appear in the next chunk.

For example:

```text
Chunk 1:
The man who did the crime was John. He was arrested yesterday.

Chunk 2:
He was arrested yesterday. The police found evidence at his house.
```

The sentence:

```text
He was arrested yesterday.
```

appears in both chunks.

### Why use overlap?

Overlap helps prevent important information from being lost at chunk boundaries.

```text
Chunk 1 ──────────────┐
                      │ overlap
                      ↓
                ────────────── Chunk 2
```

So:

> **Chunk overlap helps maintain context between neighboring chunks.**

---

# 5. Semantic Chunking

Semantic chunking divides text based on **meaning**, rather than only size or separators.

The goal is to keep related information together.

### Example

Suppose a document contains:

```text
We launched the iPhone 18 today.
It has a 200 MP camera and a new processor.

Apple's revenue increased by 18% this year.
The company's profit also increased.
```

Semantic chunking could create:

```text
Chunk 1 → iPhone 18
           - Launch
           - 200 MP camera
           - New processor

Chunk 2 → Apple Finance
           - Revenue increased by 18%
           - Profit increased
```

The chunks are organized around **topics and meaning**.

### Advantage

When the user asks:

```text
What are the specifications of the iPhone 18?
```

the retriever can find the chunk containing the iPhone information without retrieving unrelated financial information.

---

# Chunking Strategy Summary

| Strategy            | How it splits text                      | Main issue                                  |
| ------------------- | --------------------------------------- | ------------------------------------------- |
| **Fixed-size**      | Fixed words/tokens/characters           | Can break meaning                           |
| **Separator-based** | Paragraphs/sentences/etc.               | Chunk sizes can vary greatly                |
| **Recursive**       | Multiple separators + maximum size      | More complex but preserves structure better |
| **Semantic**        | Based on meaning/topic                  | More computationally expensive              |
| **Overlap**         | Repeats surrounding text between chunks | Uses more storage/tokens                    |

---

# Easy Way to Remember

Think of chunking like cutting a book:

```text
Fixed-size
→ Cut every N words.

Separator-based
→ Cut at paragraphs/sentences.

Recursive
→ Try paragraph → sentence → word
   until the chunk is small enough.

Semantic
→ Cut according to meaning/topic.

Overlap
→ Keep a little part of the previous chunk
   in the next chunk.
```

## The Main Goal of Chunking

```text
              GOOD CHUNK
                  │
        ┌─────────┴─────────┐
        ↓                   ↓
   Small enough        Large enough
   for retrieval       to preserve
                       meaning
```

**Good RAG chunking = relevant information + preserved context + manageable size.**
