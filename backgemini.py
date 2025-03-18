from flask import Flask, request, jsonify
from flask_cors import CORS
import pymupdf
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.generativeai as genai
import os
import requests

app = Flask(__name__)
CORS(app, origins=["https://athulzacharia.github.io", "http://localhost:3000"])

# Use environment variable for API key
genai.configure(api_key=os.environ.get("GEMINI_API_KEY", "AIzaSyB4WZudVc6NS00M_y9z4SuK415mLlADbNw"))

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
chat_history = []

# PDF settings
PDF_URL = "https://raw.githubusercontent.com/athulzacharia/teleretain-backend/main/temp.pdf"
PDF_PATH = "temp.pdf"
FAISS_INDEX_PATH = "faiss_index.bin"

# Download PDF if not already present
def download_pdf(url, save_path):
    if not os.path.exists(save_path):  # Avoid re-downloading
        response = requests.get(url)
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(response.content)
            print("PDF downloaded successfully.")
        else:
            print("Failed to download PDF:", response.status_code)

download_pdf(PDF_URL, PDF_PATH)

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = pymupdf.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])
    return text

# Create FAISS vector store with on-disk storage
def create_vector_store(text_chunks):
    embeddings = embed_model.encode(text_chunks, convert_to_numpy=True)
    d = embeddings.shape[1]

    index = faiss.IndexIDMap(faiss.IndexFlatL2(d))
    ids = np.arange(len(embeddings))  # Generate unique IDs
    index.add_with_ids(embeddings, ids)  # Store embeddings with IDs

    # Save index to disk
    faiss.write_index(index, FAISS_INDEX_PATH)
    print("FAISS index saved to disk.")

    return index, embeddings, text_chunks

# Load FAISS index from disk (if available)
def load_faiss_index():
    if os.path.exists(FAISS_INDEX_PATH):
        print("Loading FAISS index from disk...")
        return faiss.read_index(FAISS_INDEX_PATH)
    return None

# Load and process the PDF at startup
text = extract_text_from_pdf(PDF_PATH)
text_chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]  # Larger chunks reduce memory

index = load_faiss_index()
if index is None:
    index, embeddings, chunks = create_vector_store(text_chunks)
else:
    chunks = text_chunks  # Ensure chunks exist even if index is preloaded

# Chatbot API
@app.route("/chat", methods=["POST"])
def chat():
    user_query = request.json["query"]

    # Embed the query
    query_embedding = embed_model.encode([user_query], convert_to_numpy=True)

    # Retrieve relevant context
    D, I = index.search(query_embedding, k=3)
    retrieved_text = " ".join([chunks[i] for i in I[0]])

    # Create a prompt
    prompt = f"Context: {retrieved_text}\nUser: {user_query}\nAI:"

    # Generate response using Gemini AI
    model = genai.GenerativeModel("gemini-2.0-pro-exp")
    response = model.generate_content(prompt)
    chat_response = response.text if response.text else "Sorry, I couldn't generate a response."

    # Store chat history
    chat_history.append({"user": user_query, "ai": chat_response})

    return jsonify({"response": chat_response, "chat_history": chat_history})

# API to get chat history
@app.route("/history", methods=["GET"])
def get_chat_history():
    return jsonify({"chat_history": chat_history})

if __name__ == "__main__":
    # Use environment variable for port
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
