# Conversational PDF Chatbot (RAG + FastAPI + Telegram)

This project implements a chatbot that allows users to have conversations about the content of a specific PDF document. It uses the Retrieval-Augmented Generation (RAG) pattern, leveraging OpenAI's language models (for generating answers and embeddings) and a vector similarity search against chunks extracted from the PDF.

The core logic is exposed via a FastAPI backend, and an optional Telegram bot provides a user-friendly interface. Interactions are logged for verification purposes.

## Features

*   **Conversational Interface:** Chat naturally with the content of a PDF.
*   **Retrieval-Augmented Generation (RAG):** Answers are grounded in the actual text retrieved from the PDF, reducing hallucination.
*   **Contextual Grounding:** Retrieves relevant text snippets from the PDF to base the answer upon.
*   **Direct Quoting:** Incorporates direct quotes from the PDF into answers when relevant.
*   **Conversation History:** Maintains context across multiple turns in a conversation (handled per-user in the Telegram bot).
*   **FastAPI Backend:** Provides a robust API endpoint (`/query`) for the core RAG logic.
*   **Telegram Bot Frontend (Optional):** Makes the chatbot easily accessible via Telegram.
*   **Interaction Logging:** Logs queries, retrieved context, and generated answers to a JSON Lines file for easy review and accuracy verification.
*   **Configurable:** Uses environment variables for API keys, file paths, and other settings.

## Technology Stack

*   **Python 3.7+**
*   **Backend:**
    *   FastAPI: Modern web framework for building APIs.
    *   Uvicorn: ASGI server to run FastAPI.
    *   OpenAI Python Library: For accessing GPT models (chat completion) and embedding models.
    *   spaCy: For NLP tasks like sentence segmentation during text chunking.
    *   PyMuPDF: For extracting text content from PDF files.
    *   Numpy: For numerical operations (vector similarity).
    *   python-dotenv: For managing environment variables.
*   **Frontend (Optional):**
    *   python-telegram-bot: Library for creating Telegram bots.
    *   httpx: Asynchronous HTTP client for the bot to communicate with the FastAPI backend.

## Setup Instructions

**1. Prerequisites:**

*   Python 3.7 or higher installed.
*   `pip` (Python package installer).
*   Git (optional, for cloning the repository).
*   An **OpenAI API Key**.
*   A **Telegram Bot Token** (if using the Telegram bot). Create a bot via BotFather on Telegram.

**2. Clone the Repository (Optional):**

```bash
git clone <your-repository-url>
cd <your-repository-directory>
```

**3. Create and Activate a Virtual Environment (Recommended):**

*   **Windows:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```
*   **macOS/Linux:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

**4. Install Dependencies:**

```bash
pip install --upgrade pip
pip install "fastapi[all]" openai spacy numpy python-dotenv PyMuPDF python-telegram-bot httpx
```

**5. Download spaCy Model:**

```bash
python -m spacy download en_core_web_sm
```

**6. Prepare PDF Document:**

*   Place your PDF file in the project directory.
*   By default, the application looks for `knowledge_base.pdf`. You can change this using the `PDF_FILEPATH` environment variable (see Configuration).

## Configuration

The application uses a `.env` file to manage sensitive information and settings. Create a file named `.env` in the root directory of the project and add the following variables:

```dotenv
# --- Required ---
OPENAI_API_KEY="sk-YOUR_OPENAI_API_KEY_HERE"

# --- PDF Configuration ---
# Path to your PDF file (relative or absolute)
PDF_FILEPATH="knowledge_base.pdf"

# --- Optional: For better prompts and API description ---
BOOK_TITLE="Your Book Title Here" # e.g., "The Hitchhiker's Guide"

# --- Required ONLY if using the Telegram Bot ---
TELEGRAM_BOT_TOKEN="YOUR_TELEGRAM_BOT_TOKEN_FROM_BOTFATHER_HERE"

# --- Optional: URL for the Telegram Bot to reach the FastAPI service ---
# Default is suitable if running both on the same machine. Change if deployed separately.
# RAG_API_URL="http://localhost:8000"

# --- Optional: Logging Configuration ---
# File to log interactions (queries, context, answers)
# QUERY_LOG_FILE="query_log.jsonl"

# --- Optional: Tuning Parameters (Defaults are in the code) ---
# CHUNK_TARGET_SIZE_CHARS=1000
# RELEVANCE_THRESHOLD=0.70
# MAX_CONCURRENT_EMBEDDING_REQUESTS=50
# MAX_HISTORY_TURNS=5
```

**Important:**

*   Replace placeholders with your actual keys and desired settings.
*   Keep your `.env` file **secure** and do not commit it to version control (add `.env` to your `.gitignore` file).

## Running the Application

You need to run the FastAPI backend first. The Telegram bot is optional and runs as a separate process.

**1. Run the Backend (FastAPI Service):**

*   Make sure your virtual environment is activated.
*   Ensure your PDF file exists at the path specified by `PDF_FILEPATH` (or the default `knowledge_base.pdf`).
*   Navigate to the project's root directory in your terminal.
*   Run the Uvicorn server:

    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000
    ```
    *   Use `--reload` during development for automatic restarts on code changes (`uvicorn main:app --reload ...`).
    *   The first time you run this, it will process the PDF (extract text, chunk, generate embeddings), which might take some time depending on the PDF size and your connection to the OpenAI API. Monitor the console output.

*   You can access the API documentation (Swagger UI) at `http://localhost:8000/docs`.

**2. Run the Frontend (Telegram Bot - Optional):**

*   Make sure the FastAPI backend is running.
*   Ensure `TELEGRAM_BOT_TOKEN` is set correctly in your `.env` file.
*   Open a **new** terminal window or tab.
*   Make sure your virtual environment is activated in this new terminal.
*   Navigate to the project's root directory.
*   Run the bot script:

    ```bash
    python telegram_bot.py
    ```

*   The bot will start polling for updates from Telegram. To stop it, press `Ctrl+C`.

## Usage

**1. Using the FastAPI Endpoint:**

*   The main endpoint is `POST /query`.
*   It expects a JSON payload with:
    *   `user_input` (string): The user's question or message.
    *   `history` (list, optional): A list of previous messages in the conversation, used for context. Each message is an object `{"role": "user" | "assistant", "content": "message text"}`. For the first query, send an empty list `[]`.
*   **Example Request (First Turn):**
    ```json
    {
      "user_input": "What is the main theme of chapter 1?",
      "history": []
    }
    ```
*   **Example Request (Follow-up):**
    ```json
    {
      "user_input": "Can you elaborate on the 'uncertainty principle' mentioned?",
      "history": [
        {"role": "user", "content": "What is the main theme of chapter 1?"},
        {"role": "assistant", "content": "Chapter 1 primarily explores..."}
      ]
    }
    ```
*   The response will be a JSON object containing:
    *   `answer` (string): The generated answer from the chatbot.
    *   `retrieved_context` (list): The text snippets retrieved from the PDF used to generate the answer (useful for debugging/verification).
    *   `history` (list): The updated conversation history, including the latest user input and the generated answer. **Your client application should use this returned history for the *next* request.**

**2. Using the Telegram Bot:**

*   Find your bot on Telegram (the one you created with BotFather).
*   Send `/start` for a welcome message.
*   Simply send your questions as regular messages.
*   The bot manages conversation history automatically for your chat.
*   Use `/clear` to reset the conversation history for your chat.

## Interaction Logging

*   Successful interactions processed by the FastAPI backend (query, retrieved context, answer) are logged to the file specified by `QUERY_LOG_FILE` (default: `query_log.jsonl`).
*   Each line in this file is a valid JSON object representing one interaction turn.
*   **Log Entry Structure:**
    ```json
    {
      "timestamp": "UTC timestamp in ISO format",
      "user_input": "The user's message",
      "history_context": [ /* List of previous ChatMessage objects */ ],
      "retrieved_context": [ /* List of text snippets retrieved */ ],
      "generated_answer": "The final answer sent to the user"
    }
    ```
*   This log file is useful for verifying the accuracy of the bot's answers against the retrieved context.

## Potential Improvements

*   **Persistent History:** Replace the in-memory history dictionary in `telegram_bot.py` with a database (e.g., SQLite, PostgreSQL) or a key-value store (e.g., Redis) for persistence across bot restarts.
*   **Vector Database:** For larger documents or better performance/scalability, replace the in-memory `knowledge_base_chunks` list with a dedicated vector database (e.g., ChromaDB, FAISS with an index file, Pinecone, Weaviate). This would involve changing the embedding storage and retrieval logic.
*   **Advanced Chunking:** Explore more sophisticated text chunking strategies (e.g., recursive character splitting, semantic chunking) for potentially better retrieval results.
*   **Web UI:** Create a simple web interface using Streamlit or Gradio that interacts with the FastAPI backend.
*   **Asynchronous PDF Processing:** For very large PDFs, the initial processing at startup can be long. Move PDF parsing and embedding generation to a background task or a separate processing script.
*   **Error Handling & User Feedback:** Provide more specific error messages to the user via the Telegram bot or API.
*   **Telegram Webhooks:** For production scalability, switch the Telegram bot from polling to using webhooks. This requires hosting the bot script on a publicly accessible server with HTTPS.

## License

[Specify your license here, e.g., MIT License]
```
