import os
import logging
import numpy as np
import spacy
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import AsyncOpenAI, OpenAIError
import asyncio
from pathlib import Path
import fitz  # PyMuPDF
import json # For JSON handling
from datetime import datetime # For timestamping

# --------------------------
# Configuration & Setup
# --------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Book/PDF Configuration ---
PDF_FILEPATH = Path(os.getenv("PDF_FILEPATH", "knowledge_base.pdf"))
# ** NEW: Get book title for persona prompt **
BOOK_TITLE = os.getenv("BOOK_TITLE", "the document") # Default if not set
CHUNK_TARGET_SIZE_CHARS = 1000
RELEVANCE_THRESHOLD = 0.70
MAX_CONCURRENT_EMBEDDING_REQUESTS = 50 # Concurrency limit for embeddings
MAX_HISTORY_TURNS = 10 # Limit how many past Q&A pairs to keep in memory

# --- ** NEW: Logging Configuration ** ---
LOG_FILENAME = Path(os.getenv("QUERY_LOG_FILE", "query_log.jsonl")) # Use .jsonl extension
# Create a lock to safely write to the log file from async tasks
log_lock = asyncio.Lock()

# --- OpenAI Client ---
api_key = os.getenv("OPENAI_API_KEY")
if not api_key: raise ValueError("Missing OPENAI_API_KEY environment variable.")
try:
    client = AsyncOpenAI(api_key=api_key)
    logging.info("Successfully initialized AsyncOpenAI client.")
except OpenAIError as e:
    logging.error(f"Failed to initialize OpenAI client: {e}"); raise

# --- spaCy Model ---
try:
    nlp = spacy.load("en_core_web_sm")
    logging.info("Successfully loaded spaCy model 'en_core_web_sm'.")
except OSError:
    logging.error("spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm"); raise
except Exception as e:
    logging.error(f"An unexpected error occurred loading spaCy model: {e}"); raise

# --- FastAPI App ---
app = FastAPI(
    title=f"Chat with {BOOK_TITLE}", # Updated title
    description=f"An API to conversationally query the content of '{PDF_FILEPATH.name}'.",
    version="0.4.0"
)

# --------------------------
# Knowledge Base Store (In-Memory)
# --------------------------
knowledge_base_chunks = [] # Populated at startup

# --------------------------
# Data Models
# --------------------------
class ChatMessage(BaseModel):
    role: str # "user" or "assistant"
    content: str

class QueryRequest(BaseModel):
    user_input: str
    # ** NEW: Add history list, defaulting to empty **
    history: list[ChatMessage] = Field(default_factory=list)

class QueryResponse(BaseModel):
    answer: str
    retrieved_context: list[str] # Still useful for debugging/transparency
    # ** NEW: Return updated history **
    history: list[ChatMessage]

# --------------------------
# Utility Functions (Most remain similar, generate_response changes significantly)
# --------------------------

def extract_entities(text: str) -> list[str]:
    doc = nlp(text); entities = [ent.text for ent in doc.ents]
    logging.info(f"Extracted entities: {entities}"); return entities

async def get_embedding(text: str, model: str = "text-embedding-ada-002") -> np.ndarray | None:
    try:
        text = text.replace("\n", " "); response = await client.embeddings.create(input=[text], model=model)
        return np.array(response.data[0].embedding)
    except Exception as e:
        logging.error(f"Error generating embedding for text '{text[:50]}...': {e}"); return None

async def get_embedding_with_semaphore(text: str, semaphore: asyncio.Semaphore, model: str = "text-embedding-ada-002") -> np.ndarray | None:
    """Acquires semaphore before calling get_embedding."""
    async with semaphore:
        try: return await get_embedding(text, model)
        except Exception as e:
            logging.error(f"Error in get_embedding_with_semaphore for text '{text[:50]}...': {e}"); return None

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None: return 0.0
    norm_a = np.linalg.norm(a); norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0: return 0.0
    return np.dot(a, b) / (norm_a * norm_b)

def chunk_text_by_sentences(text: str, target_char_size: int) -> list[str]:
    doc = nlp(text); chunks = []; current_chunk = []; current_len = 0
    for sent in doc.sents:
        s_text = sent.text.strip(); s_len = len(s_text)
        if not s_text: continue
        if not current_chunk or (current_len + s_len + 1) <= target_char_size * 1.2:
            current_chunk.append(s_text); current_len += s_len + 1
        else:
            if current_chunk: chunks.append(" ".join(current_chunk))
            current_chunk = [s_text]; current_len = s_len
    if current_chunk: chunks.append(" ".join(current_chunk))
    logging.info(f"Split text into {len(chunks)} chunks."); return chunks

# --- ** NEW: Logging Function ** ---
async def log_interaction(
    user_input: str,
    history: list[ChatMessage],
    retrieved_context: list[str],
    answer: str
):
    """Appends interaction details to the JSON Lines log file safely."""
    log_entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z", # UTC timestamp in ISO format
        "user_input": user_input,
        "history_context": [msg.dict() for msg in history], # Log history *before* this turn
        "retrieved_context": retrieved_context,
        "generated_answer": answer,
    }
    async with log_lock: # Acquire the lock before file access
        try:
            with open(LOG_FILENAME, "a", encoding="utf-8") as f:
                # Dump the dictionary as a JSON string and add a newline
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            # logging.debug(f"Successfully logged interaction to {LOG_FILENAME}") # Optional debug log
        except IOError as e:
            logging.error(f"Failed to write to log file {LOG_FILENAME}: {e}")
        except Exception as e:
            logging.error(f"Unexpected error during logging: {e}")

def extract_text_from_pdf(pdf_path: Path) -> str | None:
    if not pdf_path.is_file(): logging.error(f"PDF not found: {pdf_path}"); return None
    try:
        logging.info(f"Opening PDF: {pdf_path}"); doc = fitz.open(pdf_path)
        full_text = ""; logging.info(f"Extracting text from {len(doc)} pages...")
        for page in doc: full_text += page.get_text("text", sort=True) + "\n"
        doc.close(); logging.info(f"PDF extraction complete ({len(full_text)} chars).")
        return full_text if full_text.strip() else None
    except Exception as e:
        logging.error(f"Error reading PDF {pdf_path}: {e}", exc_info=True); return None

async def initialize_knowledge_base_from_pdf():
    global knowledge_base_chunks; knowledge_base_chunks = []
    logging.info(f"Initializing knowledge base from PDF: {PDF_FILEPATH}")
    full_text = extract_text_from_pdf(PDF_FILEPATH)
    if not full_text: logging.error("Failed PDF text extraction."); return
    logging.info(f"Chunking text (~{CHUNK_TARGET_SIZE_CHARS} chars/chunk)...")
    text_chunks = chunk_text_by_sentences(full_text, CHUNK_TARGET_SIZE_CHARS)
    if not text_chunks: logging.warning("No text chunks generated."); return

    items_to_embed = [{"id": i, "text": chunk, "embedding": None} for i, chunk in enumerate(text_chunks)]
    knowledge_base_chunks.extend(items_to_embed)
    logging.info(f"Generating embeddings for {len(items_to_embed)} chunks (Concurrency: {MAX_CONCURRENT_EMBEDDING_REQUESTS})...")
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_EMBEDDING_REQUESTS)
    tasks = [get_embedding_with_semaphore(item["text"], semaphore) for item in items_to_embed]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    success, failed = 0, 0
    for i, res in enumerate(results):
        item_id = items_to_embed[i]["id"]
        if isinstance(res, Exception): logging.error(f"Embedding failed for chunk {item_id}: {res}"); failed += 1
        elif res is not None:
            for kb_item in knowledge_base_chunks:
                if kb_item["id"] == item_id: kb_item["embedding"] = res; break
            success += 1
        else: logging.error(f"Embedding failed for chunk {item_id} (returned None)."); failed += 1
    logging.info(f"KB Initialization: {success} embeddings generated, {failed} failed.")
    if success == 0 and len(items_to_embed) > 0: logging.error("CRITICAL: No embeddings generated.")


def retrieve_context(query_embedding: np.ndarray, top_k: int = 3) -> list[str]:
    if query_embedding is None or not knowledge_base_chunks: return []
    scored = []
    for item in knowledge_base_chunks:
        if item.get("embedding") is not None:
            score = cosine_similarity(query_embedding, item["embedding"])
            if score >= RELEVANCE_THRESHOLD: scored.append((score, item["text"]))
    scored.sort(key=lambda x: x[0], reverse=True)
    top_texts = [text for score, text in scored[:top_k]]
    logging.info(f"Retrieved {len(top_texts)} relevant context snippets (threshold={RELEVANCE_THRESHOLD}).")
    return top_texts

async def generate_response(
    user_input: str,
    history: list[ChatMessage],
    retrieved_context: list[str],
    model: str = "gpt-4-turbo-preview" # Or your preferred model
) -> str:
    """Generates a conversational response using persona, history, context, and encouraging quotes."""

    # --- 1. Construct the System Prompt with Persona and Quoting Instructions ---
    system_prompt = f"""You are the embodiment of the book '{BOOK_TITLE}'. Your knowledge is strictly limited to the content of this book.
    Answer the user's questions conversationally, as if you are the book speaking.
    Use the information provided under "Relevant Information" to formulate your response.

    ***IMPORTANT INSTRUCTION***: If a direct quote from the 'Relevant Information' accurately and concisely answers or strongly supports a point in your response, incorporate it. Use standard quotation marks ("...") for these quotes. Ensure the quote is *exactly* as it appears in the relevant information snippets. Do *not* invent quotes.

    NEVER mention "the context provided", "the documents", "snippets", or "the information given to me". Speak naturally from your perspective (as '{BOOK_TITLE}').
    If the relevant information doesn't contain the answer to the user's question, clearly state that the topic isn't covered within your pages. Do not make up information.
    Keep your answers concise and relevant to the query, drawing only from the provided information.
    Maintain a helpful and knowledgeable tone consistent with '{BOOK_TITLE}'. Be concise and to the point, when you are not quoting from the relevant information.
    """

    # --- 2. Prepare Conversation History ---
    truncated_history = history[-(MAX_HISTORY_TURNS * 2):]

    # --- 3. Format Retrieved Context ---
    if retrieved_context:
        context_text = "\n\n".join([f"Snippet {i+1}:\n{text}" for i, text in enumerate(retrieved_context)])
        # Integrate context more naturally into the prompt for the *last* user message
        context_for_prompt = f"""Relevant Information from my pages:
        ---
        {context_text}
        ---
        Based *only* on the information above (and potentially using direct quotes from it), answer the following question: {user_input}
        """
    else:
        context_for_prompt = f"""Relevant Information from my pages:
        ---
        [No specific information found for this query.]
        ---
        Based *only* on the fact that no specific information was found, answer the following question: {user_input}
        """

    # --- 4. Construct the Full Prompt Messages ---
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend([{"role": msg.role, "content": msg.content} for msg in truncated_history])
    messages.append({"role": "user", "content": context_for_prompt}) # Add the user query *with* context info

    # --- 5. Call OpenAI API ---
    try:
        logging.info(f"Generating response with {len(messages)} messages (prompt includes quoting instruction).")
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.25, # Keep temperature relatively low for faithfulness to text/quotes
            max_tokens=350 # Allow slightly longer responses if quotes are included
        )
        answer = response.choices[0].message.content.strip()
        logging.info("Successfully generated AI response.")
        return answer
    except OpenAIError as e:
        logging.error(f"OpenAI API error during chat completion: {e}")
        return f"I encountered an issue trying to formulate a response based on '{BOOK_TITLE}'. Please try again."
    except Exception as e:
        logging.error(f"Unexpected error during chat completion: {e}")
        return "Sorry, an unexpected internal error occurred."


# --------------------------
# FastAPI Events
# --------------------------
@app.on_event("startup")
async def startup_event():
    logging.info("Application startup: Initializing resources...")
    await initialize_knowledge_base_from_pdf()
    logging.info("Application startup complete.")

# --------------------------
# API Endpoints
# --------------------------
@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Handles conversational queries using RAG and history."""
    user_input = request.user_input.strip()
    history = request.history # Get history from request
    if not user_input:
        raise HTTPException(status_code=400, detail="Input query cannot be empty.")

    logging.info(f"Received query: '{user_input}' with {len(history)} history messages.")

    # --- RAG Pipeline ---
    # 1. Embed Query (Async)
    # Consider embedding a modified query that includes recent history context? (Advanced)
    query_embedding = await get_embedding(user_input)
    if query_embedding is None:
        logging.error("Failed to generate query embedding.")
        raise HTTPException(status_code=500, detail="Could not process query embedding.")

    # 2. Retrieve Context (Sync)
    retrieved_context = retrieve_context(query_embedding, top_k=3)

    # 3. Generate Response (Async) - Pass history
    answer = await generate_response(user_input, history, retrieved_context)

        # --- ** NEW: Log the interaction AFTER getting the answer ** ---
    await log_interaction(
        user_input=user_input,
        history=history, # Pass the history *before* this turn
        retrieved_context=retrieved_context,
        answer=answer
    )
    # --- End Logging ---

    # --- Update History ---
    # Add current user query and assistant answer
    updated_history = history + [
        ChatMessage(role="user", content=user_input),
        ChatMessage(role="assistant", content=answer),
    ]
    # Keep history trimmed (optional, as generate_response also truncates input)
    updated_history = updated_history[-(MAX_HISTORY_TURNS * 2):]

    return QueryResponse(
        answer=answer,
        retrieved_context=retrieved_context, # Keep for potential client-side display
        history=updated_history # Return the new history
    )

@app.get("/", include_in_schema=False)
async def root():
    return {"message": f"Chat API for '{BOOK_TITLE}' is running. POST to /query. See /docs."}

# --------------------------
# Entry Point
# --------------------------
if __name__ == "__main__":
    import uvicorn
    if "PDF_FILEPATH" not in os.environ and not PDF_FILEPATH.is_file():
         logging.warning(f"Default PDF '{PDF_FILEPATH}' not found. Set PDF_FILEPATH env var or create the file.")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)