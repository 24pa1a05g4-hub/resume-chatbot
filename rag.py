import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from google import genai
from dotenv import load_dotenv  # <--- This is the new tool

# 1. Setup
# Load the .env file
load_dotenv()

# Get the key from the file
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ ERROR: Could not find GEMINI_API_KEY in .env file!")
else:
    client = genai.Client(api_key=api_key)

# 2. Model Configuration
# We use the model that works for your account
model = SentenceTransformer("all-MiniLM-L6-v2")

# 3. Load Data
try:
    with open("documents.txt", "r") as f:
        documents = [line.strip() for line in f if line.strip()]
except FileNotFoundError:
    documents = []
    print("WARNING: documents.txt not found! Please create it.")

# 4. Create Index
if documents:
    embeddings = model.encode(documents)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
else:
    index = None

def get_answer(user_query):
    if not index:
        return "I have no knowledge base yet. Please create documents.txt."

    # Search
    query_vec = model.encode([user_query])
    distances, indices = index.search(query_vec, k=2)
    
    # Retrieve
    retrieved_text = "\n".join([documents[i] for i in indices[0]])
    
    # Generate
    prompt = f"""
    You are an assistant for Alex Turing. Answer based ONLY on the context below.
    Context: {retrieved_text}
    Question: {user_query}
    """
    
    try:
        response = client.models.generate_content(model="gemini-1.5-flash", contents=prompt)
        return response.text
    except Exception as e:
        return f"Google Error: {str(e)}"
