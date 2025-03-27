import logging
import os
import httpx # Async HTTP client
from dotenv import load_dotenv
from telegram import Update, constants # Import constants for ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

# --------------------------
# Configuration
# --------------------------
load_dotenv() # Load environment variables from .env

# --- Telegram Bot Token ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_TOKEN:
    raise ValueError("Missing TELEGRAM_BOT_TOKEN environment variable.")

# --- FastAPI Service URL ---
RAG_API_URL = os.getenv("RAG_API_URL", "http://localhost:8000") # Default to local if not set
QUERY_ENDPOINT = f"{RAG_API_URL}/query"

# --- In-Memory Conversation History Store ---
# Simple dictionary to store history per chat_id
# Warning: This is lost on bot restart. For persistence, use a database or file.
conversation_histories = {}

# --- Logging Setup ---
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING) # Reduce httpx verbosity
logger = logging.getLogger(__name__)

# --------------------------
# Bot Command Handlers
# --------------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a welcome message when the /start command is issued."""
    user = update.effective_user
    welcome_message = (
        f"Hello {user.mention_html()}! 👋\n\n"
        f"I can help you explore the content of the configured document. "
        f"Just ask me anything about it!\n\n"
        f"Use /clear to reset our conversation history."
    )
    await update.message.reply_html(welcome_message)
    # Optionally clear history on /start
    chat_id = update.effective_chat.id
    if chat_id in conversation_histories:
        del conversation_histories[chat_id]
        logger.info(f"Cleared history for chat_id {chat_id} on /start")


async def clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Clears the conversation history for the current chat."""
    chat_id = update.effective_chat.id
    if chat_id in conversation_histories:
        del conversation_histories[chat_id]
        logger.info(f"Cleared history for chat_id {chat_id}")
        await update.message.reply_text("Our conversation history has been cleared.")
    else:
        await update.message.reply_text("We don't have any conversation history yet.")


# --------------------------
# Message Handler (Calls RAG API)
# --------------------------

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles regular text messages and interacts with the RAG API."""
    chat_id = update.effective_chat.id
    user_input = update.message.text

    if not user_input:
        return # Ignore empty messages

    logger.info(f"Received message from chat_id {chat_id}: '{user_input}'")

    # --- Show "typing..." status ---
    await context.bot.send_chat_action(chat_id=chat_id, action=constants.ChatAction.TYPING)

    # --- Retrieve current history for this chat ---
    current_history = conversation_histories.get(chat_id, [])
    logger.debug(f"History for chat_id {chat_id} before API call: {len(current_history)} messages")


    # --- Prepare payload for FastAPI ---
    payload = {
        "user_input": user_input,
        "history": current_history
        # FastAPI expects history as list of {'role': str, 'content': str}
    }

    # --- Call the FastAPI RAG service ---
    async with httpx.AsyncClient(timeout=60.0) as client: # Increased timeout for potentially slow RAG
        try:
            response = await client.post(QUERY_ENDPOINT, json=payload)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            # --- Process successful response ---
            api_data = response.json()
            answer = api_data.get("answer", "Sorry, I couldn't get a proper answer.")
            new_history = api_data.get("history", []) # Get updated history from API

            # Store the updated history back
            conversation_histories[chat_id] = new_history
            logger.debug(f"History for chat_id {chat_id} after API call: {len(new_history)} messages")

            # Send the answer back to Telegram
            await update.message.reply_text(answer)

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error calling RAG API: {e.response.status_code} - {e.response.text}")
            error_message = "Sorry, I encountered an issue processing your request. The service might be unavailable."
            # Try to get more detail from FastAPI error if available
            try:
                error_detail = e.response.json().get("detail")
                if error_detail:
                     error_message += f"\nDetails: {error_detail}"
            except Exception:
                pass # Ignore if parsing error detail fails
            await update.message.reply_text(error_message)

        except httpx.RequestError as e:
            logger.error(f"Network error calling RAG API: {e}")
            await update.message.reply_text("Sorry, I couldn't connect to the knowledge service. Please check if it's running and try again.")

        except Exception as e:
            logger.exception(f"An unexpected error occurred in handle_message for chat_id {chat_id}:")
            await update.message.reply_text("An unexpected error occurred while processing your message.")


# --------------------------
# Main Bot Application Setup
# --------------------------

def main() -> None:
    """Starts the Telegram bot."""
    logger.info("Starting Telegram Bot...")

    # Create the Application and pass it your bot's token.
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # --- Register handlers ---
    # On different commands - answer in Telegram
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("clear", clear))
    # You can add a /help command similarly

    # On non-command i.e message - handle the message using handle_message
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Run the bot until the user presses Ctrl-C
    logger.info("Bot is polling for updates...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

    logger.info("Bot stopped.")


if __name__ == "__main__":
    main()