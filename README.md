# Implementing LangGraph Supervisor Multi-Agent System with A2A Protocol

## Overview

![Overview](https://storage.googleapis.com/zenn-user-upload/18c2620bb9dc-20250924.png)

This repository demonstrates how to implement a Supervisor-type multi-agent system using the A2A Protocol, originally created with LangGraph.
We first explain how to build a Supervisor-type multi-agent system using `langgraph-supervisor`, then reimplement the same system using the A2A Protocol.

https://github.com/a2aproject/A2A

## Motivation

While StrandsAgent and Google ADK provide A2A Protocol modules as libraries, LangGraph does not yet have such modules.
Additionally, there are still few examples of implementing supervisor-type multi-agent systems. By demonstrating the implementation method with LangGraph, we aim to provide a reference for other developers building similar systems.

## Sample Code

The code introduced in this article is published on GitHub.

https://github.com/5enxia/langgraph-multiagent-with-a2a

## Implementing Supervisor-Type Multi-Agent with `langgraph-supervisor`

### Environment Setup

We use `uv` for package management.
The author implemented this using `python 3.13`.

https://github.com/astral-sh/uv

First, install the necessary libraries:

```sh
uv add langchain langchain-google-genai langchain-openai langgraph langgraph-supervisor a2a-sdk[http-server]
```

Next, set the following environment variables.
Use your preferred LLM.
The code examples use the `gpt-4.1` family.

```sh
export OPENAI_API_KEY="your_openai_api_key"
export GOOGLE_GENAI_API_KEY="your_google_genai_api_key"
```

### Supervisor-Type Multi-Agent Implementation

Here's a code example implementing a Supervisor-type multi-agent using LangGraph:

```python
# langgraph-only/__main__.py
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain.chat_models import init_chat_model

model = init_chat_model(model='gpt-4.1-nano')
memory = MemorySaver()

@tool
def get_exchange_rate():
    """Get the exchange rate between USD and JPY."""
    return "1ドル = 147円"


currency_agent = create_react_agent(
    model=model,
    tools=[get_exchange_rate],
    checkpointer=memory,
    name="currency_agent",
    prompt=(
        'You answer questions about currency exchange rates. '
        'Use the tool to get the exchange rate between USD and JPY.'
    )
)


@tool
def get_weather(location: str) -> str:
    """Get the current weather."""
    return f"{location} is Sunny"


weather_agent = create_react_agent(
    model=model,
    tools=[get_weather],
    checkpointer=memory,
    name="weather_agent",
    prompt=(
        "You answer questions about the weather in a given location."
        "Use the tool to get the current weather."
    )
)


workflow = create_supervisor(
    model=model,
    agents=[
        currency_agent,
        weather_agent,
    ],
    checkpointer=memory,
    prompt=(
        "You are a supervisor managing two agents:"
        "Assign work to one agent at a time, do not call agents in parallel."
        "Do not do any work yourself."
    )
)

app = workflow.compile()
content = "How much is 1 USD in JPY?"
# content = "What's the weather in Tokyo?"
result = app.invoke({"messages": [HumanMessage(content=content)]})
for message in result["messages"]:
    print(message)
```

This code uses `langgraph_supervisor` to build a Supervisor-type multi-agent system.
There are two agents: `currency_agent` and `weather_agent`, responsible for currency conversion and weather information respectively.
The Supervisor assigns tasks to the appropriate agent based on user questions.
※You can switch the commented lines to see which agent gets assigned the task.

### Execution

Save the above code as `langgraph-only/__main__.py` and run it with:

```sh
uv run langgraph-only
```

## Reimplementation with A2A Protocol

Next, we'll reimplement the same Supervisor-type multi-agent system using the A2A Protocol.

### AgentExecutor

In the A2A Protocol, you need to inherit from the `AgentExecutor` class to manage agent execution.
Here's an example of the `LangGraphAgentExecutor` class for using LangGraph agents with the A2A Protocol:

```python
# a2a/sub_agents/common/agent_executor.py
import logging

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    InternalError,
    InvalidParamsError,
    Part,
    TaskState,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils import (
    new_agent_text_message,
    new_task,
)
from a2a.utils.errors import ServerError

from .adapter import LangGraphAgentAdapter


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LangGraphAgentExecutor(AgentExecutor):
    """LangGraph AgentExecutor Example."""

    def __init__(self, adapter: LangGraphAgentAdapter):
        self.adapter = adapter

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        error = self._validate_request(context)
        if error:
            raise ServerError(error=InvalidParamsError())

        query = context.get_user_input()
        task = context.current_task
        if not task:
            task = new_task(context.message)  # type: ignore
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        try:
            async for item in self.adapter.stream(query, task.context_id):
                is_task_complete = item['is_task_complete']
                require_user_input = item['require_user_input']

                if not is_task_complete and not require_user_input:
                    await updater.update_status(
                        TaskState.working,
                        new_agent_text_message(item['content']),
                    )
                elif require_user_input:
                    await updater.update_status(
                        TaskState.input_required,
                        new_agent_text_message(item['content']),
                        require_user_input=True,
                    )
                    break

        except Exception as e:
            logger.error(f'An error occurred while streaming the response: {e}')
            raise ServerError(error=InternalError()) from e

    def _validate_request(self, context: RequestContext) -> bool:
        return False

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        raise ServerError(error=UnsupportedOperationError())
```

This class inherits from the A2A Protocol's `AgentExecutor` and manages LangGraph agent execution.
The `execute` method receives input from users (or potentially other agents), passes it to the LangGraph agent for processing.
Depending on the processing progress, it sends messages or updates task status (`working`, `input_required`, `completed`, etc.).
The `cancel` method defines handling for task cancellation.
Since we don't specifically support cancel operations here, it returns an `UnsupportedOperationError`.

#### Adapter

The Adapter provides an interface between LangGraph agents and the A2A Protocol.
This class is a custom implementation by the author that converts LangGraph agent messages to A2A Protocol Task format.

```python
# a2a/sub_agents/common/adapter.py
from collections.abc import AsyncIterable
from typing import Any, Literal
from pydantic import BaseModel

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import Runnable

FORMAT_INSTRUCTION = (
    'Set response status to input_required if the user needs to provide more information to complete the request.'
    'Set response status to error if there is an error while processing the request.'
    'Set response status to completed if the request is complete.'
)

class ResponseFormat(BaseModel):
    """Respond to the user in this format."""

    status: Literal['input_required', 'completed', 'error'] = 'input_required'
    message: str


class LangGraphAgentAdapter:
    def __init__(self, agent: Runnable):
        self.graph = agent

    async def stream(self, query, context_id) -> AsyncIterable[dict[str, Any]]:
        inputs = {'messages': [('user', query)]}
        config = {'configurable': {'thread_id': context_id}}

        for item in self.graph.stream(inputs, config, stream_mode='values'):
            message = item['messages'][-1]
            if (
                isinstance(message, AIMessage)
                and hasattr(message, 'tool_calls')
                and len(message.tool_calls) > 0
            ):
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': f"Using tool: {message.tool_calls[0]['name']}"
                }
            elif isinstance(message, ToolMessage):
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': f"Tool result: {message.content}"
                }

        yield self.get_agent_response(config)

    def get_agent_response(self, config):
        current_state = self.graph.get_state(config)
        structured_response = current_state.values.get('structured_response')
        if structured_response and isinstance(
            structured_response, ResponseFormat
        ):
            if structured_response.status == 'input_required':
                return {
                    'is_task_complete': False,
                    'require_user_input': True,
                    'content': structured_response.message,
                }
            if structured_response.status == 'error':
                return {
                    'is_task_complete': True,
                    'require_user_input': False,
                    'content': structured_response.message,
                }
            if structured_response.status == 'completed':
                return {
                    'is_task_complete': True,
                    'require_user_input': False,
                    'content': structured_response.message,
                }

        return {
            'is_task_complete': False,
            'require_user_input': True,
            'content': (
                'We are unable to process your request at the moment. '
                'Please try again.'
            ),
        }

    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']
```

### Sub-Agent Implementation

Next, we'll implement the sub-agents (`currency_agent` and `weather_agent`) as independent A2A Servers.

#### Currency Agent

```python
# a2a/sub_agents/currency_agent.py
import logging

# client/server
import httpx
import uvicorn

# A2A
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import (
    BasePushNotificationSender,
    InMemoryPushNotificationConfigStore,
    InMemoryTaskStore,
)
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

# LangChain/LangGraph
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import init_chat_model

# Custom
from common.agent_executor import LangGraphAgentExecutor
from common.adapter import LangGraphAgentAdapter, ResponseFormat, FORMAT_INSTRUCTION


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

host = '0.0.0.0'
port = 10000


@tool
def get_exchange_rate():
    """Get the exchange rate between USD and JPY."""
    return "1 USD = 147円"


agent = create_react_agent(
    model=init_chat_model(model='gpt-4.1-nano'),
    tools=[get_exchange_rate],
    checkpointer=MemorySaver(),
    prompt=(
        'You are a specialized assistant for currency conversions. '
        "Your sole purpose is to use the 'get_exchange_rate' tool to answer questions about currency exchange rates. "
        'If the user asks about anything other than currency conversion or exchange rates, '
        'politely state that you cannot help with that topic and can only assist with currency-related queries. '
        'Do not attempt to answer unrelated questions or use tools for other purposes.'
    ),
    response_format=(FORMAT_INSTRUCTION, ResponseFormat)
)

skill = AgentSkill(
    id='convert_currency',
    name='Currency Exchange Rates Tool',
    description='Helps with exchange values between various currencies',
    tags=['currency conversion', 'currency exchange'],
    examples=['What is exchange rate between USD and GBP?'],
)

agent_card = AgentCard(
    name='Currency Agent',
    description='Helps with exchange rates for currencies',
    url=f'http://{host}:{port}/',
    version='1.0.0',
    default_input_modes=LangGraphAgentAdapter.SUPPORTED_CONTENT_TYPES,
    default_output_modes=LangGraphAgentAdapter.SUPPORTED_CONTENT_TYPES,
    capabilities=AgentCapabilities(streaming=True, push_notifications=True),
    skills=[skill],
)

httpx_client = httpx.AsyncClient()
push_config_store = InMemoryPushNotificationConfigStore()
push_sender = BasePushNotificationSender(httpx_client=httpx_client,
                config_store=push_config_store)


adapter = LangGraphAgentAdapter(agent=agent)
request_handler = DefaultRequestHandler(
    agent_executor=LangGraphAgentExecutor(adapter=adapter),
    task_store=InMemoryTaskStore(),
    push_config_store=push_config_store,
    push_sender=push_sender
)
server = A2AStarletteApplication(
    agent_card=agent_card, http_handler=request_handler
)

uvicorn.run(server.build(), host=host, port=port)
```

#### Weather Agent

```python
# a2a/sub_agents/weather_agent.py
import logging

# client/server
import httpx
import uvicorn

# A2A
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import (
    BasePushNotificationSender,
    InMemoryPushNotificationConfigStore,
    InMemoryTaskStore,
)
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

# LangChain/LangGraph
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import init_chat_model

# Custom
from common.agent_executor import LangGraphAgentExecutor
from common.adapter import LangGraphAgentAdapter, ResponseFormat, FORMAT_INSTRUCTION


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

host = '0.0.0.0'
port = 20000


@tool
def get_weather(city_name: str) -> str:
    """Get the current weather."""
    return f"{city_name} is Sunny"


agent = create_react_agent(
    model=init_chat_model(model='gpt-4.1-nano'),
    tools=[get_weather],
    checkpointer=MemorySaver(),
    prompt=(
        'You are a specialized assistant for weather information.'
        "Your sole purpose is to use the 'get_weather' tool to answer questions about weather conditions. "
        'If the user asks about anything other than weather information, '
        'politely state that you cannot help with that topic and can only assist with weather-related queries. '
        'Do not attempt to answer unrelated questions or use tools for other purposes.'
    ),
    response_format=(
        FORMAT_INSTRUCTION,
        ResponseFormat
    ),
)

skill = AgentSkill(
    id='get_weather',
    name='Get Weather',
    description='Fetch the current weather for a location',
    tags=['weather', 'location'],
    examples=['What is the weather like in New York?', 'Tell me the weather in San Francisco'],
)

agent_card = AgentCard(
    name='Weather Expert',
    description='Fetch the current weather for a location',
    url=f'http://{host}:{port}/',
    version='1.0.0',
    default_input_modes=LangGraphAgentAdapter.SUPPORTED_CONTENT_TYPES,
    default_output_modes=LangGraphAgentAdapter.SUPPORTED_CONTENT_TYPES,
    capabilities=AgentCapabilities(streaming=True, push_notifications=True),
    skills=[skill],
)

httpx_client = httpx.AsyncClient()
push_config_store = InMemoryPushNotificationConfigStore()
push_sender = BasePushNotificationSender(httpx_client=httpx_client,
                config_store=push_config_store)

adapter = LangGraphAgentAdapter(agent=agent)
request_handler = DefaultRequestHandler(
    agent_executor=LangGraphAgentExecutor(adapter=adapter),
    task_store=InMemoryTaskStore(),
    push_config_store=push_config_store,
    push_sender=push_sender
)
server = A2AStarletteApplication(
    agent_card=agent_card, http_handler=request_handler
)

uvicorn.run(server.build(), host=host, port=port)
```

After creating the LangGraph agent using `create_react_agent`, we define `AgentSkill` and `AgentCard`.
`AgentSkill` and `AgentCard` are used to provide agent information (overview, skills, input/output types) according to A2A Protocol specifications.
After setting up the http client and Push Notification configurations, we build the agent server using `A2AStarletteApplication` and start it with `uvicorn`.

### Supervisor Agent Implementation

Finally, we'll implement the Supervisor agent.

#### A2AClientToolProvider

Before defining the agent, we'll create a client to connect to sub-agents using the A2A Client.
This is based on the Strands Agents A2A Client.

This class provides tools for retrieving AgentCards from sub-agents and sending messages to sub-agents that have successfully retrieved AgentCards.

```python
import asyncio
import logging
from typing import Any
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory, A2AClient
from a2a.types import AgentCard, Message, Part, PushNotificationConfig, Role, TextPart
from langchain_core.tools import tool
from langchain_core.tools import BaseTool as AgentTool, StructuredTool

DEFAULT_TIMEOUT = 300  # set request timeout to 5 minutes

logger = logging.getLogger(__name__)


class A2AClientToolProvider:
    """A2A Client tool provider that manages multiple A2A agents and exposes synchronous tools."""

    def __init__(
        self,
        known_agent_urls: list[str] | None = None,
        timeout: int = DEFAULT_TIMEOUT,
        webhook_url: str | None = None,
        webhook_token: str | None = None,
    ):
        """
        Initialize A2A client tool provider.

        Args:
            known_agent_urls: List of A2A agent URLs to use (defaults to None)
            timeout: Timeout for HTTP operations in seconds (defaults to 300)
            webhook_url: Optional webhook URL for push notifications
            webhook_token: Optional authentication token for webhook notifications
        """
        self.timeout = timeout
        self._known_agent_urls: list[str] = known_agent_urls or []
        self._discovered_agents: dict[str, AgentCard] = {}
        self._httpx_client: httpx.AsyncClient | None = None
        self._client_factory: ClientFactory | None = None
        self._initial_discovery_done: bool = False

        # Push notification configuration
        self._webhook_url = webhook_url
        self._webhook_token = webhook_token
        self._push_config: PushNotificationConfig | None = None

        if self._webhook_url and self._webhook_token:
            self._push_config = PushNotificationConfig(
                id=f"webhook-{uuid4().hex[:8]}", url=self._webhook_url, token=self._webhook_token
            )

    @property
    def tools(self) -> list[AgentTool]:
        """Extract all @tool decorated methods from this instance."""
        _tools = [
            self.a2a_discover_agent,
            self.a2a_list_discovered_agents,
            self.a2a_send_message
        ]

        tools = [
            StructuredTool.from_function(coroutine=tool_func)
            for tool_func in _tools
        ]
        return tools

    async def _ensure_httpx_client(self) -> httpx.AsyncClient:
        """Ensure the shared HTTP client is initialized."""
        if self._httpx_client is None:
            self._httpx_client = httpx.AsyncClient(timeout=self.timeout)
        return self._httpx_client

    async def _ensure_client_factory(self) -> ClientFactory:
        """Ensure the ClientFactory is initialized."""
        if self._client_factory is None:
            httpx_client = await self._ensure_httpx_client()
            config = ClientConfig(
                httpx_client=httpx_client,
                streaming=False,
                push_notification_configs=[self._push_config] if self._push_config else [],
            )
            self._client_factory = ClientFactory(config)
        return self._client_factory

    async def _create_a2a_card_resolver(self, url: str) -> A2ACardResolver:
        """Create a new A2A card resolver for the given URL."""
        httpx_client = await self._ensure_httpx_client()
        logger.info(f"A2ACardResolver created for {url}")
        return A2ACardResolver(httpx_client=httpx_client, base_url=url)

    async def _discover_known_agents(self) -> None:
        """Discover all agents provided during initialization."""

        async def _discover_agent_with_error_handling(url: str):
            """Helper method to discover an agent with error handling."""
            try:
                await self._discover_agent_card(url)
            except Exception as e:
                logger.error(f"Failed to discover agent at {url}: {e}")

        tasks = [_discover_agent_with_error_handling(url) for url in self._known_agent_urls]
        if tasks:
            await asyncio.gather(*tasks)

        self._initial_discovery_done = True

    async def _ensure_discovered_known_agents(self) -> None:
        """Ensure initial discovery of agent URLs from constructor has been done."""
        if not self._initial_discovery_done and self._known_agent_urls:
            await self._discover_known_agents()

    async def _discover_agent_card(self, url: str) -> AgentCard:
        """Internal method to discover and cache an agent card."""
        if url in self._discovered_agents:
            return self._discovered_agents[url]

        resolver = await self._create_a2a_card_resolver(url)
        agent_card = await resolver.get_agent_card()
        self._discovered_agents[url] = agent_card
        logger.info(f"Successfully discovered and cached agent card for {url}")

        return agent_card

    async def a2a_discover_agent(self, url: str) -> dict[str, Any]:
        """
        Discover an A2A agent and return its agent card with capabilities.

        This function fetches the agent card from the specified A2A agent URL
        and caches it for future use.

        Args:
            url: The base URL of the A2A agent to discover

        Returns:
            dict: Discovery result including:
                - success: Whether the operation succeeded
                - agent_card: The discovered agent card (if successful)
                - error: Error message (if failed)
                - url: The agent URL that was queried
        """
        return await self._discover_agent_card_tool(url)

    async def _discover_agent_card_tool(self, url: str) -> dict[str, Any]:
        """Internal async implementation for discover_agent_card tool."""
        try:
            await self._ensure_discovered_known_agents()
            agent_card = await self._discover_agent_card(url)
            return {
                "status": "success",
                "agent_card": agent_card.model_dump(mode="python", exclude_none=True),
                "url": url,
            }
        except Exception as e:
            logger.exception(f"Error discovering agent card for {url}")
            return {
                "status": "error",
                "error": str(e),
                "url": url,
            }

    async def a2a_list_discovered_agents(self) -> dict[str, Any]:
        """
        List all discovered A2A agents and their capabilities.

        Returns:
            dict: Information about all discovered agents including:
                - success: Whether the operation succeeded
                - agents: List of discovered agent cards
                - total_count: Total number of discovered agents
        """
        return await self._list_discovered_agents()

    async def _list_discovered_agents(self) -> dict[str, Any]:
        """Internal async implementation for list_discovered_agents."""
        try:
            await self._ensure_discovered_known_agents()
            agents = [
                agent_card.model_dump(mode="python", exclude_none=True)
                for agent_card in self._discovered_agents.values()
            ]
            return {
                "status": "success",
                "agents": agents,
                "total_count": len(agents),
            }
        except Exception as e:
            logger.exception("Error listing discovered agents")
            return {
                "status": "error",
                "error": str(e),
                "total_count": 0,
            }

    async def a2a_send_message(
        self, message_text: str, target_agent_url: str, message_id: str | None = None
    ) -> dict[str, Any]:
        """
        Send a message to a specific A2A agent and return the response.

        Args:
            message_text: The message content to send to the agent
            target_agent_url: The URL of the target A2A agent
            message_id: Optional message ID for tracking (generates UUID if not provided)

        Returns:
            dict: Response data including:
                - success: Whether the message was sent successfully
                - response: The agent's response message (if successful)
                - error: Error message (if failed)
                - message_id: The message ID used for tracking
                - target_agent_url: The agent URL that was contacted
        """
        return await self._send_message(message_text, target_agent_url, message_id)

    async def _send_message(
        self, message_text: str, target_agent_url: str, message_id: str | None = None
    ) -> dict[str, Any]:
        """Internal async implementation for send_message."""

        try:
            await self._ensure_discovered_known_agents()

            # Get the agent card and create client using factory
            agent_card = await self._discover_agent_card(target_agent_url)
            client_factory = await self._ensure_client_factory()
            client = client_factory.create(agent_card)

            if message_id is None:
                message_id = uuid4().hex

            message = Message(
                kind="message",
                role=Role.user,
                parts=[TextPart(text=message_text)],
                message_id=message_id,
            )

            logger.info(f"Sending message to {target_agent_url}")

            # With streaming=False, this will yield exactly one result
            async for event in client.send_message(message):
                if isinstance(event, Message):
                    response_content = ""
                    for part in event.parts:
                        if hasattr(part, 'text'):
                            response_content += part.text

                    return {
                        "status": "success",
                        "response": response_content,
                        "message_id": message_id,
                        "target_agent_url": target_agent_url,
                    }

            # This should never be reached with streaming=False
            return {
                "status": "error",
                "error": "No response received from agent",
                "message_id": message_id,
                "target_agent_url": target_agent_url,
            }

        except Exception as e:
            logger.exception(f"Error sending message to {target_agent_url}")
            return {
                "status": "error",
                "error": str(e),
                "message_id": message_id,
                "target_agent_url": target_agent_url,
            }
```

We provide three asynchronous methods as LangChain Tools: `a2a_discover_agent`, `a2a_list_discovered_agents`, and `a2a_send_message`.
Since the `@tool` decorator cannot be used with instance methods, we wrap them using `StructuredTool.from_function` in the `tools` property.

https://github.com/langchain-ai/langchain/discussions/9404

#### Supervisor Agent

Finally, we implement the Supervisor agent.
Using the three tools defined in `A2AClientToolProvider` (`a2a_discover_agent`, `a2a_list_discovered_agents`, `a2a_send_message`), we connect to sub-agents and delegate tasks.

```python
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
async def main():
    response = await supervisor.ainvoke({
        "messages": [
            {
                "role": "user",
                "content": "What's the weather in Tokyo?"
                # "content": "How much is 1 USD in JPY?"
            }
        ]
    })
    messages = response['messages']
    for message in messages:
        print(message.content)

asyncio.run(main())
```

※Similar to the implementation without A2A Protocol, you can switch the commented user messages to see which agent gets assigned the task.

### Running the Application

Save the above code as `a2a/supervisor/__main__.py` and run with the following commands:

```sh
uv run a2a/sub_agents/currency_agent
```

```sh
uv run a2a/sub_agents/weather_agent
```

```sh
uv run a2a/supervisor
```

When executed, the Supervisor agent starts up, accesses each sub-agent's `/.well-known/agent-card.json` to retrieve AgentCards, then assigns tasks to the appropriate sub-agent based on user questions and returns results. You can see logs of this process.

## Summary

In this article, we explained how to build multi-agent systems using LangGraph.
We particularly focused on implementing agent-to-agent communication using the A2A protocol.
We demonstrated how a Supervisor agent coordinates with various sub-agents to provide appropriate information in response to user requests.

We hope this will serve as a reference for those developing loosely coupled multi-agent systems combining LangGraph and the A2A protocol in the future.

## Referenced Code

### Supervisor-Type Multi-Agent

https://github.com/langchain-ai/langgraph-supervisor-py

### Adapter Implementation

https://github.com/a2aproject/a2a-samples/blob/main/samples/python/agents/langgraph/app/agent.py

### AgentExecutor Implementation

https://github.com/a2aproject/a2a-samples/blob/main/samples/python/agents/langgraph/app/agent_executor.py

### A2A Client

https://strandsagents.com/latest/documentation/docs/user-guide/concepts/multi-agent/agent-to-agent/#strands-a2a-tool

https://github.com/strands-agents/tools/blob/main/src/strands_tools/a2a_client.py
