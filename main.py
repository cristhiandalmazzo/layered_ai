import os
import logging
import numpy as np
import spacy
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import AsyncOpenAI, OpenAIError
import asyncio
from pathlib import Path
import fitz  # PyMuPDF library for PDF handling

# --------------------------
# Configuration & Setup
# --------------------------

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- PDF Configuration ---
# Use environment variable PDF_FILEPATH or default to 'knowledge_base.pdf'
PDF_FILEPATH = Path(os.getenv("PDF_FILEPATH", "knowledge_base.pdf"))
CHUNK_TARGET_SIZE_CHARS = 1000
RELEVANCE_THRESHOLD = 0.70

# --- Concurrency Limiting ---
# Max number of simultaneous embedding requests to OpenAI
MAX_CONCURRENT_EMBEDDING_REQUESTS = 50 # ADJUST AS NEEDED (start lower, e.g., 20-50)

# --- OpenAI Client ---
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Missing OPENAI_API_KEY environment variable.")
try:
    client = AsyncOpenAI(api_key=api_key)
    logging.info("Successfully initialized AsyncOpenAI client.")
except OpenAIError as e:
    logging.error(f"Failed to initialize OpenAI client: {e}")
    raise

# --- spaCy Model ---
try:
    nlp = spacy.load("en_core_web_sm")
    logging.info("Successfully loaded spaCy model 'en_core_web_sm'.")
except OSError:
    logging.error("spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
    raise
except Exception as e:
    logging.error(f"An unexpected error occurred loading spaCy model: {e}")
    raise

# --- FastAPI App ---
app = FastAPI(
    title="RAG MVP: PDF-Based AI Chat Assistant",
    description="API using a PDF file as a knowledge base via RAG.",
    version="0.3.0"
)

# --------------------------
# Knowledge Base Store (In-Memory)
# --------------------------
knowledge_base_chunks = []

# --------------------------
# Data Models
# --------------------------
class QueryRequest(BaseModel):
    user_input: str

class QueryResponse(BaseModel):
    answer: str
    retrieved_context: list[str]
    entities: list[str]

# --------------------------
# Utility Functions (extract_entities, get_embedding, cosine_similarity - remain the same)
# --------------------------
def extract_entities(text: str) -> list[str]:
    """Extract named entities (sync)."""
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]
    logging.info(f"Extracted entities: {entities}")
    return entities

async def get_embedding(text: str, model: str = "text-embedding-ada-002") -> np.ndarray | None:
    """Generate embedding using OpenAI (async)."""
    try:
        text = text.replace("\n", " ")
        response = await client.embeddings.create(input=[text], model=model)
        embedding = response.data[0].embedding
        return np.array(embedding)
    except OpenAIError as e:
        logging.error(f"OpenAI API error generating embedding for text '{text[:50]}...': {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error generating embedding: {e}")
        return None

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity (sync)."""
    if a is None or b is None: return 0.0
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0: return 0.0
    return np.dot(a, b) / (norm_a * norm_b)

def chunk_text_by_sentences(text: str, target_char_size: int) -> list[str]:
    """Chunks text by grouping sentences (sync)."""
    doc = nlp(text)
    chunks = []
    current_chunk_sentences = []
    current_chunk_len = 0
    for sent in doc.sents:
        sentence_text = sent.text.strip()
        if not sentence_text: continue
        sentence_len = len(sentence_text)
        if not current_chunk_sentences or (current_chunk_len + sentence_len + 1) <= target_char_size * 1.2:
            current_chunk_sentences.append(sentence_text)
            current_chunk_len += sentence_len + 1
        else:
            if current_chunk_sentences: chunks.append(" ".join(current_chunk_sentences))
            current_chunk_sentences = [sentence_text]
            current_chunk_len = sentence_len
    if current_chunk_sentences: chunks.append(" ".join(current_chunk_sentences))
    logging.info(f"Split text into {len(chunks)} chunks.")
    return chunks

def extract_text_from_pdf(pdf_path: Path) -> str | None:
    """Extracts text content from a PDF file using PyMuPDF."""
    if not pdf_path.is_file():
        logging.error(f"PDF file not found at: {pdf_path}")
        return None
    try:
        logging.info(f"Opening PDF file: {pdf_path}")
        doc = fitz.open(pdf_path)
        full_text = ""
        logging.info(f"Extracting text from {len(doc)} pages...")
        for page_num, page in enumerate(doc):
            # Extract text as plain text blocks
            page_text = page.get_text("text", sort=True) # 'sort=True' can help with reading order
            if page_text:
                full_text += page_text + "\n" # Add newline between pages for some separation
        doc.close()
        logging.info(f"PDF text extraction complete ({len(full_text)} characters).")
        if not full_text.strip():
            logging.warning(f"Extracted text from PDF {pdf_path} is empty.")
            return None
        return full_text
    except fitz.EmptyFileError:
        logging.error(f"ERROR: PDF file at {pdf_path} seems to be empty or corrupted.")
        return None
    # PyMuPDF raises RuntimeError for password errors by default
    except RuntimeError as e:
        if "password" in str(e):
             logging.error(f"ERROR: PDF file {pdf_path} is password protected.")
        else:
             logging.error(f"Runtime error processing PDF {pdf_path}: {e}", exc_info=True)
        return None
    except Exception as e:
        logging.error(f"Error opening or reading PDF {pdf_path}: {e}", exc_info=True)
        return None


# Define a helper function to wrap the embedding call with the semaphore
async def get_embedding_with_semaphore(text: str, semaphore: asyncio.Semaphore, model: str = "text-embedding-ada-002") -> np.ndarray | None:
    """Acquires semaphore before calling get_embedding."""
    async with semaphore:
        # Optional: Add a small delay to be nicer to the API endpoint, might help avoid rate limits
        # await asyncio.sleep(0.05)
        try:
            # Call the original embedding function
            embedding = await get_embedding(text, model)
            return embedding
        except Exception as e:
            # Log error here as well, as the original get_embedding might return None silently on API errors
            logging.error(f"Error in get_embedding_with_semaphore for text '{text[:50]}...': {e}")
            return None


async def initialize_knowledge_base_from_pdf():
    """
    Reads PDF, extracts text, chunks it, computes embeddings (with concurrency limit),
    and populates the knowledge base.
    """
    global knowledge_base_chunks
    knowledge_base_chunks = []

    logging.info(f"Initializing knowledge base from PDF: {PDF_FILEPATH}")

    # Step 1: Extract text
    full_text = extract_text_from_pdf(PDF_FILEPATH)
    if not full_text:
        logging.error("Failed to extract text from PDF. Initialization aborted.")
        return

    # Step 2: Chunk text
    logging.info(f"Chunking extracted text with target size ~{CHUNK_TARGET_SIZE_CHARS} chars...")
    text_chunks = chunk_text_by_sentences(full_text, CHUNK_TARGET_SIZE_CHARS)
    if not text_chunks:
        logging.warning("No text chunks generated from PDF content.")
        return

    # Step 3: Generate embeddings with concurrency control
    items_to_embed = [{"id": i, "text": chunk, "embedding": None} for i, chunk in enumerate(text_chunks)]
    knowledge_base_chunks.extend(items_to_embed) # Add items to global list

    logging.info(f"Generating embeddings for {len(items_to_embed)} chunks with max concurrency {MAX_CONCURRENT_EMBEDDING_REQUESTS}...")

    # Create the semaphore
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_EMBEDDING_REQUESTS)

    # Create tasks using the throttled wrapper function
    tasks = [
        get_embedding_with_semaphore(item["text"], semaphore) # Pass semaphore to the wrapper
        for item in items_to_embed
    ]

    # Run tasks concurrently (gather manages them, semaphore limits active API calls)
    results = await asyncio.gather(*tasks, return_exceptions=True) # return_exceptions=True is safer

    successful_embeddings = 0
    failed_count = 0
    for i, result in enumerate(results):
        item_id = items_to_embed[i]["id"] # Get corresponding item id

        if isinstance(result, Exception):
            logging.error(f"Failed to generate embedding for chunk {item_id} due to exception: {result}")
            failed_count += 1
        elif result is not None:
             # Find the item in knowledge_base_chunks by id and update its embedding
            for kb_item in knowledge_base_chunks:
                if kb_item["id"] == item_id:
                    kb_item["embedding"] = result # Assign the numpy array
                    break
            successful_embeddings += 1
        else:
            # get_embedding_with_semaphore might return None if get_embedding itself failed internally
            logging.error(f"Failed to generate embedding for chunk {item_id} (returned None).")
            failed_count += 1


    logging.info(f"Knowledge base initialization complete. {successful_embeddings} embeddings generated, {failed_count} failed.")
    if successful_embeddings == 0 and len(items_to_embed) > 0:
        logging.error("CRITICAL: No embeddings generated. Check OpenAI connectivity/key & previous logs.")


def retrieve_context(query_embedding: np.ndarray, top_k: int = 3) -> list[str]:
    """Retrieve relevant chunks based on similarity (sync)."""
    if query_embedding is None or not knowledge_base_chunks:
        return []
    scored_entries = []
    for item in knowledge_base_chunks:
        kb_embedding = item.get("embedding")
        if kb_embedding is not None:
            score = cosine_similarity(query_embedding, kb_embedding)
            if score >= RELEVANCE_THRESHOLD:
                 scored_entries.append((score, item["text"]))
    scored_entries.sort(key=lambda x: x[0], reverse=True)
    top_entries_text = [text for score, text in scored_entries[:top_k]]
    logging.info(f"Retrieved {len(top_entries_text)} relevant context snippets (top_k={top_k}, threshold={RELEVANCE_THRESHOLD}).")
    return top_entries_text

async def generate_response(user_input: str, entities: list, retrieved_context: list, model: str = "gpt-4-turbo-preview") -> str:
    """Generate response using LLM (async)."""
    context_text = "\n".join([f"- {text}" for text in retrieved_context]) if retrieved_context else "No relevant context found in the provided PDF content."
    prompt_messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer the user's query based *primarily* on the provided context from the PDF document. If the context doesn't contain the answer, state that the information isn't available in the provided text. Be concise."},
        {"role": "user", "content": f"""Based *only* on the following context snippets from a PDF document, please answer the query. Do not use outside knowledge.

Context:
{context_text}

User Query: {user_input}
"""}
    ]
    try:
        response = await client.chat.completions.create(
            model=model, messages=prompt_messages, temperature=0.2, max_tokens=200
        )
        answer = response.choices[0].message.content.strip()
        logging.info("Successfully generated AI response.")
        return answer
    except OpenAIError as e:
        logging.error(f"OpenAI API error during chat completion: {e}")
        return "Sorry, I encountered an error communicating with the AI model."
    except Exception as e:
        logging.error(f"Unexpected error during chat completion: {e}")
        return "Sorry, an unexpected error occurred."

# --------------------------
# FastAPI Events
# --------------------------

@app.on_event("startup")
async def startup_event():
    """Run async initialization on startup (using PDF)."""
    logging.info("Application startup: Initializing resources...")
    await initialize_knowledge_base_from_pdf() # Changed to PDF initializer
    logging.info("Application startup complete.")

# --------------------------
# API Endpoints (query_endpoint, root - remain the same structure)
# --------------------------

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Handle user queries using the PDF-based RAG pipeline."""
    user_input = request.user_input.strip()
    if not user_input:
        raise HTTPException(status_code=400, detail="Input query cannot be empty.")
    logging.info(f"Received query: '{user_input}'")

    entities = extract_entities(user_input)
    query_embedding = await get_embedding(user_input)
    if query_embedding is None:
        logging.error("Failed to generate query embedding.")
        raise HTTPException(status_code=500, detail="Could not process query embedding.")

    retrieved_context = retrieve_context(query_embedding, top_k=3)
    answer = await generate_response(user_input, entities, retrieved_context)

    return QueryResponse(
        answer=answer,
        retrieved_context=retrieved_context,
        entities=entities
    )

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "PDF RAG API is running. POST queries to /query. See /docs."}

# --------------------------
# Entry Point
# --------------------------

if __name__ == "__main__":
    import uvicorn
    if "PDF_FILEPATH" not in os.environ and not PDF_FILEPATH.is_file():
         logging.warning(f"Default PDF file '{PDF_FILEPATH}' not found.")
         logging.warning("Create the file or set the PDF_FILEPATH environment variable.")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)