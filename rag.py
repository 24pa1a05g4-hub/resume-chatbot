import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from google import genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# 1. Setup
model = SentenceTransformer("all-MiniLM-L6-v2")
client = genai.Client()

# 2. Load Data
try:
    with open("documents.txt", "r") as f:
        documents = [line.strip() for line in f if line.strip()]
except FileNotFoundError:
    documents = []
    print("WARNING: documents.txt not found! Create it to add knowledge.")

# 3. Create Index (Only if documents exist)
if documents:
    embeddings = model.encode(documents)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
else:
    index = None

def get_answer(user_query):
    if not index:
        return "I have no knowledge base yet. Please create documents.txt."

    # 1. Search
    query_vec = model.encode([user_query])
    distances, indices = index.search(query_vec, k=2)
    
    # 2. Retrieve
    retrieved_text = "\n".join([documents[i] for i in indices[0]])
    
    # 3. Generate
    prompt = f"""
    You are an assistant for Alex Turing. Answer based ONLY on the context below.
    Context: {retrieved_text}
    Question: {user_query}
    """
    response = client.models.generate_content(model='gemini-2.0-flash', contents=prompt)
    return response.text