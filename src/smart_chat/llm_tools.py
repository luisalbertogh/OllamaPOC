import logging

from langchain_core.tools import tool

# Configure logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_selected_tool(tool_call: dict):
    """Select the tool to invoke based on AI tool call.

    Args:
        tool_call (dict): The tool call retrieved from the AI model.

    Returns:
        The tool to invoke.
    """
    return {"multiply": multiply, "add": add}[tool_call["name"].lower()]


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


@tool
def add(a: int, b: int) -> int:
    """Add two numbers.

    Args:
        a (int): Number to multiply.
        b (int): Numer to multiply.

    Returns:
        int: The result of the multiplication.
    """
    logger.info('In add...')
    return a + b
