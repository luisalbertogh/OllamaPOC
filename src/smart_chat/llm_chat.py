import logging
from typing import Any, Dict

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from smart_chat.llm_tools import get_selected_tool

# Configure logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SmartChatWrapper:
    """Class to itnitialize and run LLM chat.
    """

    def __init__(self, model_name: str, base_url: str, temperature: float = None, tools: list = []):
        """Initialize the class.
        """
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature

        # Initialize the ChatOllama instance
        self.llm = ChatOllama(
            model=self.model_name,
            temperature=self.temperature,
            base_url=self.base_url
        )

        # Bind tools
        if tools:
            self.llm = self.llm.bind_tools(tools)

            # Initial prompt messages
            self.messages = [
                (
                    "system",
                    "Analyse the given prompt and call a tool only if asked, if not try to give a response."
                ),
                (
                    "human",
                    "{input}"
                ),
                (
                    "placeholder",
                    "{tool_msgs}"
                )
            ]

            # Add system message to steer response
            self.prompt = ChatPromptTemplate.from_messages(self.messages)
        # No tools
        else:
            # Add system message to steer response
            self.prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a helpful assistant. Try to give a response."
                    ),
                    (
                        "human",
                        "{input}"
                    )
                ]
            )

        # Chain model with prompt
        self.chain = self.prompt | self.llm

    @property
    def params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters.

        This information is used by the LangChain callback system, which
        is used for tracing purposes make it possible to monitor LLMs.
        """
        return {
            # The model name allows users to specify custom token counting
            # rules in LLM monitoring applications (e.g., in LangSmith users
            # can provide per token pricing for their model and monitor
            # costs for the given LLM.)
            "model_name": self.model_name,
            "base_url": self.base_url,
            "temperature": self.temperature
        }

    def invoke(self, message: str) -> str:
        """Invoke the chat.

        Args:
            message (str): User messages.

        Returns:
            str: Chat reponse.
        """
        ai_msg = self.chain.invoke(message)
        return ai_msg.content

    def tool_invoke(self, message: str):
        """Invoke the chat to run tools.

        Args:
            message (str): User messages.
        """
        tool_msgs = []
        ai_msg = self.chain.invoke({"input": message})
        tool_msgs.append(ai_msg)

        # Iterate tool calls
        for tool_call in ai_msg.tool_calls:
            selected_tool = get_selected_tool(tool_call)
            tool_msg = selected_tool.invoke(tool_call)
            tool_msgs.append(tool_msg)

        # Invoke model to get interpreted tool response
        logger.debug(tool_msgs)
        print(self.chain.invoke({"input": message, "tool_msgs": tool_msgs}).content, end="", flush=True)

    def stream(self, message: str):
        """Invoke the chat using streams.

        Args:
            message (str): User message.
        """
        # Stream messages
        for chunk in self.chain.stream({"input": message}):
            print(f"{chunk.content}", end="", flush=True)
