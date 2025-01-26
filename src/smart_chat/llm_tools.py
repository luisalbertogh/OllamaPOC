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
    return {"multiply": multiply, "add": add, "learn": learn}[tool_call["name"].lower()]


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


@tool
def learn(file_path: str) -> str:
    """Load text content from a file and return it.

    Args:
        file_path (str): Path to the file with the content to learn.

    Retuns:
        str: The content of the file.
    """
    logger.info('In learn...')
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError as fnf_error:
        logger.error(f'File not found: {str(fnf_error)}.')
        return f'Calling tool with arguments {file_path} returned the following error: {type(fnf_error)}: {fnf_error}'


# List of tools to bind
tools_to_bind = [multiply, add, learn]
