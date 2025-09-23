import asyncio
import logging
from langgraph.prebuilt import create_react_agent
from a2a_client import A2AClientToolProvider
from langchain.chat_models import init_chat_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create A2A client tool provider with known agent URLs
provider = A2AClientToolProvider(known_agent_urls=[
    "http://0.0.0.0:10000", # Currency Expert
    "http://0.0.0.0:20000", # Weather Expert
])

# Create agent with A2A client tools
supervisor = create_react_agent(
    model=init_chat_model(model='gpt-4.1-mini'),
    tools=provider.tools,
    name="supervisor",
    prompt="You are a team supervisor managing a currency agent and a weather information agent."
)

# The agent can now discover and interact with A2A servers
# Standard usage

async def main():
    response = await supervisor.ainvoke({
        "messages": [
            {
                "role": "user",
                "content": "今日の東京の天気は？"
                # "content": "1ドルは為替で何円ですか？"
            }
        ]
    })
    messages = response['messages']
    for message in messages:
        print(message.content)

asyncio.run(main())