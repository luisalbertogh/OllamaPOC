# Ollama POC
# ====================
import asyncio

from langchain_ollama import ChatOllama

# Ollama chat instance
llm = ChatOllama(
    model="llama3",
    temperature=0,
    base_url="http://localhost:11434"
)


def basic_sample():
    """Basic chat sample
    """
    messages = [
        ("system", "You are a helpful translator. Translate the user sentence to French."),
        ("human", "I love programming."),
    ]
    ai_msg = llm.invoke(messages)
    print(ai_msg.content)


async def async_stream_sample():
    """Asynchrous stream chat sample
    """
    messages = [
        ("human", "Return the words Hello World!"),
    ]
    async for chunk in llm.astream_events(messages, version="v1"):
        print(chunk)


def stream_sample():
    """Stream chat sample
    """
    messages = [
        ("human", "Return the words Hello World!"),
    ]
    for chunk in llm.stream(messages):
        print(chunk.content)


if __name__ == "__main__":
    # Call async sample
    asyncio.run(async_stream_sample())
    # stream_sample()
    # basic_sample()
