import logging

from langchain_core.tools import tool

# Configure logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers.

    Args:
        a (int): Number to multiply.
        b (int): Numer to multiply.

    Returns:
        int: The result of the multiplication.
    """
    logger.info('In multiply...')
    return a * b


# Registered tools
tools = [multiply]
