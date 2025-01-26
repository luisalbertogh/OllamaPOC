import logging
import sys

from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache

from smart_chat.llm_chat import SmartChatWrapper
from smart_chat.llm_tools import tools_to_bind

# Configure logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Smart chat
smart_chat = None


def invoke(message: str):
    """Invoke the chat.

    Args:
        message (str): Input message.
    """
    # Invoke the chat
    print("ChatBot >> ", end="")
    smart_chat.tool_invoke(message)
    print("")


def init_chat():
    """Main function.
    """
    global smart_chat

    # Init chat wrapper for tool calling with predefined tools
    logger.info("Starting smart chat...")
    smart_chat = SmartChatWrapper(model_name="llama3.1",
                                  base_url="http://localhost:11434",
                                  tools=tools_to_bind)

    # Use in-memory cache up to 100 items
    set_llm_cache(InMemoryCache(maxsize=100))


# Entry point of the program
if __name__ == "__main__":
    # Init smart chat with LLM
    init_chat()

    # Infinite loop
    while True:
        # Get input message
        input_message = input("Enter your message (/bye for exit) >> ")

        # If input message is "/bye", then exit the chat
        if input_message == "/bye":
            logger.info("Exiting the chat...")
            sys.exit(0)

        # Invoke the chat
        invoke(input_message)
