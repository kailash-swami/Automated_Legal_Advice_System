from concurrent.futures import ThreadPoolExecutor
import ray
import numpy as np
import logging
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from faiss import IndexFlatL2


# Initialize Ray for parallel processing
ray.init()

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load legal documents
logging.info("Loading legal documents from 'data' directory...")
loader = DirectoryLoader("data", glob="./*.txt")
documents = loader.load()

# Debugging the loaded content
for doc in documents:
    content = doc.page_content
    logging.info(f"Document loaded: {doc.metadata.get('source', 'Unknown')}")
    logging.info(f"Content preview: {content[:200]}")  # First 200 characters

if not documents:
    logging.error("No documents loaded. Ensure 'data' contains valid text files.")
else:
    logging.info(f"Successfully loaded {len(documents)} documents.")

# Text splitting and preprocessing
logging.info("Splitting and tagging legal documents...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)  # Smaller chunks for legal texts

texts = []
for document in documents:
    text_content = document.page_content
    if not text_content.strip():
        logging.warning(f"Document {document.metadata.get('source', 'Unknown')} is empty. Skipping...")
        continue
    source_label = "Constitution" if "constitution" in document.metadata.get("source", "").lower() else "IPC"
    chunks = text_splitter.split_text(text_content)
    texts.extend([{"text": chunk, "source": source_label} for chunk in chunks])
    logging.info(f"Document split into {len(chunks)} chunks.")

# Load HuggingFace embedding model
logging.info("Loading embeddings model...")
embeddings_model = HuggingFaceEmbeddings(model_name="law-ai/InLegalBERT")

# Define embedding function
def embedding_function(text):
    return embeddings_model.embed_query(text)  # FAISS requires embeddings as NumPy arrays

# Initialize FAISS index
logging.info("Initializing FAISS index...")
index = IndexFlatL2(768)  # Set dimensionality to match the embedding model

# Prepare docstore and index-to-docstore mapping
docstore = {}
index_to_docstore_id = {}

# Parallel embedding generation
logging.info("Generating embeddings in parallel...")
def generate_embedding(text_data):
    text = text_data["text"]
    if not text.strip():
        return None
    embedding = embedding_function(text)
    return {"embedding": embedding, "text": text}

with ThreadPoolExecutor(max_workers=4) as executor:  # Adjust max_workers based on CPU
    results = list(executor.map(generate_embedding, texts))

# Filter and add results to docstore and FAISS
embeddings_list = []
for i, result in enumerate(results):
    if result is None:
        logging.warning(f"Skipping empty or invalid chunk at index {i}.")
        continue
    embeddings_list.append(result["embedding"])
    docstore[i] = result["text"]
    index_to_docstore_id[i] = i

# Convert embeddings list to a NumPy array
embeddings_array = np.array(embeddings_list)

# Add embeddings to the FAISS index
index.add(embeddings_array)  # Add embeddings all at once

# Initialize FAISS database
faiss_db = FAISS(embedding_function, index, docstore, index_to_docstore_id)

# Save the FAISS index
try:
    logging.info("Saving FAISS database...")
    faiss_db.save_local("legal_embed_db")
    logging.info("FAISS database saved successfully.")
except Exception as e:
    logging.error(f"Error saving FAISS database: {e}")

# Complete the process
logging.info("FAISS database creation completed successfully.")

# Shutdown Ray
ray.shutdown()
