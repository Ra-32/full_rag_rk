
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import os
load_dotenv()
import warnings
warnings.filterwarnings("ignore", message="IProgress not found")
groq_api_key=os.getenv("GROQ_API_KEY")
qdrant_url=os.getenv("QDRANT_URL")
qdrant_api_key=os.getenv("QDRANT_API_KEY")

if not groq_api_key and qdrant_api_key and qdrant_url:
    print("No api key found in .env")
else:
    print("Api loaded sucessfully")

client=QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key
)

print("Connected to the Qdrant cloud")

COLLECTION_NAME="knowledege"
EMBEDDING_SIZE=384

#delete collection if is already exists

if client.collection_exists(COLLECTION_NAME):
    print(f"Deleting existing collection:{COLLECTION_NAME}")
    client.delete_collection(COLLECTION_NAME)

#CREATE CONNECTION
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=EMBEDDING_SIZE,
        distance=Distance.COSINE,
    ),
)

print(f"created collection:{COLLECTION_NAME}")
print(f"Vector size:{EMBEDDING_SIZE}")
print(f"Distance.Cosine")

# load our konwledge 
with open("knowledege.txt","r",encoding="utf-8") as f:
    documents=[
        line.strip()
        for line in f
        if line.strip()
    ]

print(f"loaded {len(documents)} documents")


# create embedding

print("loaded embedding model")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("embedding model is ready")

embeddings=model.encode(documents)

print(f"Genrated {len(embeddings)} embeddings")
print(f"embedding size:{len(embeddings[0])}")


# create qdrant points

points =[]
for i ,embeddings in enumerate(embeddings):
    point=PointStruct(
        id=i+1,
        vector=embeddings.tolist(),
        payload={
            "text":documents[i]
        }
    )

    points.append(point)

# upload to qdrant 
client.upsert(
    collection_name=COLLECTION_NAME,
    points=points
)

print(f"uploaded {len(points)} documents to qdrant")

# serach the qdrant

def search(query,top_k=3):
    query_vector=model.encode(query).tolist()

    # serach qdrant to similar to these 

    results=client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True
    ).points

    return results


# test search

query="what is python"

results=search(query,top_k=3)

print("\n Serachresults:")

for result in results:
    print(f"Score:{result.score:.3f}")
    print(result.payload["text"])
    print()

# connect to groq

groq_client=Groq(api_key=groq_api_key) 


def ask_llm(question,context):
    prompt=f""" Answer the question using only the information provided below 
    content:{context} questions:{question}

    if the answer is not present in the context says:
    'i don't know based on the provided information '
    """
    
    response=groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role":"user",
                "content":prompt           
             }
        ]
    )
    return response.choices[0].message.content

# complete the pipeline

question="what is advatage of python"

result=search(question,top_k=3)

# extract the text
context="\n".join(
    result.payload["text"]
    for result in results
)

answer=ask_llm(question,context)

print("\n Final answer")

print(answer)

    