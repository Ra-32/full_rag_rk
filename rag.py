import os
from groq import Groq
from dotenv import load_dotenv
import numpy as np
from sentence_transformers import SentenceTransformer
import sys
# Load variables from .env file
load_dotenv()

# Get Groq API key
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")

print("API key found")


client = Groq(api_key=api_key)
# Load embedding model
gmodel="openai/gpt-oss-120b"

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded successfully")

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

docs=[
    "Python is a high-level, interpreted programming language.",
    "Python is widely used in web development, data science, artificial intelligence, machine learning, automation, and software development.",
    "Python was created by Guido van Rossum and first released in 1991.",

    "One of the main advantages of Python is its simple and readable syntax.",

    "Popular Python libraries include NumPy, Pandas, Scikit-learn, TensorFlow, PyTorch, and Streamlit.",
    "Retrieval-Augmented Generation, commonly known as RAG, is a technique used to improve Large Language Model applications.",

    "A RAG system retrieves relevant information from external documents before generating an answer.",

    "The typical RAG pipeline contains several steps.",

    "First, documents are loaded into the system.",

    "Second, the documents are divided into smaller chunks.",

    "Third, an embedding model converts the text chunks into numerical vectors.",

    "Fourth, these vectors are stored in a vector database.",

    "When a user asks a question, the system converts the question into an embedding and searches for similar document chunks.",

    "The retrieved documents are then provided to a Large Language Model as context.",

    "The Large Language Model uses this context to generate a more accurate answer."
]



print("Documents loaded successfully",len(docs))

embeddings = model.encode(docs)

print("document embedding size is:",sys.getsizeof(embeddings))
print("Embeddings generated successfully",embeddings.shape)


def retrieve_docs(query, docs, embeddings, top_k=3):
    similarities = []
    query_embedding = model.encode([query])[0] # query embedding 
    for doc_embedding in embeddings:
        similarity = cosine_similarity(query_embedding, doc_embedding)
        similarities.append(similarity) 


    # Highest similarity indices
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    results = []

    for i in top_indices:
        results.append({
            "document": docs[i],
            "similarity": similarities[i]
        })

    return results



def ask_llm(query, docs, embeddings, top_k=None):
    query_embedding = model.encode([query])[0]
    top_k_docs = retrieve_docs(query, docs, embeddings, top_k=top_k) if top_k is not None else []
    print("Top K retrieved documents:", top_k_docs)
    context = "\n".join([doc["document"] for doc in top_k_docs])

    print("Retrieved context:", context)
    
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    
    response = client.chat.completions.create(
        model=gmodel,
        messages=[
            {"role": "user", "content": prompt}
        ],
        
    )
    
    return response.choices[0].message.content.strip()

query = "what is the advatange of python"

response = ask_llm(query, docs, embeddings,top_k=3)
print("Response from LLM:is ------------", response)

#  traditional rag --> chunk ->embed-> cosine similartity-> retrive
# vectorless rag pageIndexrag -> build tree->llm reason over tree-> retrive exact section 



