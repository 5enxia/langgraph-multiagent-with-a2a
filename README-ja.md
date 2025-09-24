# langgraph-supervisorで作成したSupervisor型マルチエージェントをA2A Protocolを用いて再実装

## 概要

![Overview](https://storage.googleapis.com/zenn-user-upload/18c2620bb9dc-20250924.png)

このリポジトリでは、LangGraphを使用して作成したSupervisor型マルチエージェントをA2A Protocolを用いて実装する方法を示します。
はじめに、`langgraph-supervisor`を用いたSupervisor型マルチエージェントシステムの構築方法を説明し、その後、A2A Protocolを用いて同様のシステムを再実装します。

https://github.com/a2aproject/A2A

## モチベーション

StrandsAgentやGoogle ADKではA2A Protocolを用いたモジュールがライブラリとして提供されていますが、LangGraphにはそのようなモジュールがまだありません。
また、supervisor型マルチエージェントを実装した例もまだ少なく、LangGraphでの実装方法を示すことで、他の開発者が同様のシステムを構築する際の参考になると考え、取り組みました。

## サンプルコード

本記事で紹介しているコードはGitHubで公開しています。

https://github.com/5enxia/langgraph-multiagent-with-a2a

## `langgraph-supervisor`を用いたSupervisor型マルチエージェントの実装

### 環境設定

前提として、パッケージ管理として`uv`を使用しています。
また、筆者は`python 3.13`を使用して実装しています。

https://github.com/astral-sh/uv

まず、必要なライブラリをインストールします。

```sh
uv add langchain langchain-google-genai langchain-openai langgraph langgraph-supervisor a2a-sdk[http-server]
```

次に、以下の環境変数を設定します。
LLMはお好みのものを使用してください。
コード例では`gpt-4.1`ファミリーを使用しています。

```sh
export OPENAI_API_KEY="your_openai_api_key"
export GOOGLE_GENAI_API_KEY="your_google_genai_api_key"
```

### Supervisor型マルチエージェントの実装

以下に、LangGraphを使用してSupervisor型マルチエージェントを実装するコード例を示します。

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
content = "1ドルは何円ですか？"
# content = "東京の天気は？"
result = app.invoke({"messages": [HumanMessage(content=content)]})
for message in result["messages"]:
    print(message)
```

このコードでは、`langgraph_supervisor`を使用してSupervisor型マルチエージェントシステムを構築しています。
`currency_agent`と`weather_agent`の2つのエージェントがあり、それぞれ通貨換算と天気情報の提供を担当します。
Supervisorは、ユーザーからの質問に基づいて適切なエージェントにタスクを割り当てます。
※コメントアウトされている部分を切り替えることで、どちらのエージェントにタスクが割り当てられるかを確認できます。

### 実行

上記のコードを`langgraph-only/__main__.py`として保存し、以下のコマンドで実行します。

```sh
uv run langgraph-only
```

## A2A Protocolを用いた再実装

次に、A2A Protocolを用いて同様のSupervisor型マルチエージェントシステムを再実装します。

### AgentExecutor

A2A Protocolでは、エージェントの実行を管理するために`AgentExecutor`というクラスを継承して使用する必要があります。
以下に、LangGraphエージェントをA2A Protocolで使用するための`LangGraphAgentExecutor`クラスの例を示します。

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
                        new_agent_text_message(
                            item['content'],
                            task.context_id,
                            task.id,
                        ),
                    )
                elif require_user_input:
                    await updater.update_status(
                        TaskState.input_required,
                        new_agent_text_message(
                            item['content'],
                            task.context_id,
                            task.id,
                        ),
                        final=True,
                    )
                    break
                else:
                    await updater.add_artifact(
                        [Part(root=TextPart(text=item['content']))],
                        name='conversion_result',
                    )
                    await updater.complete()
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

このクラスは、A2A Protocolの`AgentExecutor`を継承し、LangGraphエージェントの実行を管理します。
`execute`メソッドでは、ユーザー(場合によってはエージェント)からの入力を受け取り、LangGraphエージェントに渡して処理を行います。
処理の進行状況に応じて、メッセージの送信またはタスクの状態を更新（`working`、`input_required`、`completed`など）します。
タスクがキャンセルされた場合の処理は`cancel`メソッドで定義しています。
ここでは、特にキャンセル操作をサポートしていないため、`UnsupportedOperationError`を返しています。

#### Adapter

Adapterは、LangGraphエージェントとA2A Protocolの間のインターフェースを提供します。
このクラスは、筆者がカスタム実装したものでLangGraphのエージェントのメッセージをA2A ProtocolのTaskの形式に変換する役割を果たします。

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
                and message.tool_calls
                and len(message.tool_calls) > 0
            ):
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': 'Looking up the exchange rates...',
                }
            elif isinstance(message, ToolMessage):
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': 'Processing the exchange rates..',
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
                    'is_task_complete': False,
                    'require_user_input': True,
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

### サブエージェントの実装

次に、サブエージェント（`currency_agent`と`weather_agent`）をそれぞれ独立したA2A Serverとして実装していきます。

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

冒頭で`create_react_agent`を使用してLangGraphエージェントを作成した後、`AgentSkill`, `AgentCard`を定義しています。
`AgentSkill`, `AgentCard`は、A2A Protocolの仕様に基づいてエージェント（概要、スキル、入出力の型）の情報提供のために使用されます。
その他、httpクライアントやPush Notificationの設定をした後、`A2AStarletteApplication`を使用してエージェントサーバーを構築し、`uvicorn`で起動します。

### スーパーバイザーエージェントの実装

最後に、Supervisorエージェントを実装していきます。

#### A2AClientToolProvider

エージェントを定義する前に、A2A Clientを使用してサブエージェントに接続するためのクライアントを作成します。
こちらは、Strands AgentsのA2A Clientを参考にしています。

このクラスはサブエージェントへのAgentCardの取得やAgentCardが取得できたサブエージェントへのメッセージ送信やの取得を行うためのツールを提供します。

```python
import asyncio
import logging
from typing import Any
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import AgentCard, Message, Part, PushNotificationConfig, Role, TextPart
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
                id=f"langgraph-webhook-{uuid4().hex[:8]}", url=self._webhook_url, token=self._webhook_token
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
                streaming=False,  # Use non-streaming mode for simpler response handling
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


    # @tool
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
                - agent_card: The full agent card data (if successful)
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


    # @tool
    async def a2a_list_discovered_agents(self) -> dict[str, Any]:
        """
        List all discovered A2A agents and their capabilities.

        Returns:
            dict: Information about all discovered agents including:
                - success: Whether the operation succeeded
                - agents: List of discovered agents with their details
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


    # @tool
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
                - response: The agent's response data (if successful)
                - error: Error message (if failed)
                - message_id: The message ID used
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
                parts=[Part(TextPart(kind="text", text=message_text))],
                message_id=message_id,
            )

            logger.info(f"Sending message to {target_agent_url}")

            # With streaming=False, this will yield exactly one result
            async for event in client.send_message(message):
                if isinstance(event, Message):
                    # Direct message response
                    return {
                        "status": "success",
                        "response": event.model_dump(mode="python", exclude_none=True),
                        "message_id": message_id,
                        "target_agent_url": target_agent_url,
                    }
                elif isinstance(event, tuple) and len(event) == 2:
                    # (Task, UpdateEvent) tuple - extract the task
                    task, update_event = event
                    return {
                        "status": "success",
                        "response": {
                            "task": task.model_dump(mode="python", exclude_none=True),
                            "update": (
                                update_event.model_dump(mode="python", exclude_none=True) if update_event else None
                            ),
                        },
                        "message_id": message_id,
                        "target_agent_url": target_agent_url,
                    }
                else:
                    # Fallback for unexpected response types
                    return {
                        "status": "success",
                        "response": {"raw_response": str(event)},
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

LangChainのToolとして、`a2a_discover_agent`, `a2a_list_discovered_agents`, `a2a_send_message`の3つの非同期メソッドを提供しています。
`@tool`デコレータをインスタンスメソッドで使用することはできないため、`tools`プロパティで`StructuredTool.from_function`を使用してラップしています。

https://github.com/langchain-ai/langchain/discussions/9404

#### Supervisor Agent

最後に、Supervisorエージェントを実装します。
`A2AClientToolProvider`で定義した、`a2a_discover_agent`, `a2a_list_discovered_agents`, `a2a_send_message`の3つのツールを使用して、サブエージェントに接続、タスクを委譲します。

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
                "content": "今日の東京の天気は？"
                # "content": "1ドルは何円？"
            }
        ]
    })
    messages = response['messages']
    for message in messages:
        print(message.content)

asyncio.run(main())
```

※A2A Protocolを使わずに実装した時と同様に、コメントアウトされているユーザメッセージを切り替えることで、どちらのエージェントにタスクが割り当てられるかを確認できます。

### 実行

上記のコードを`a2a/supervisor/__main__.py`として保存し、以下のコマンドで実行します。

```sh
uv run a2a/currency_agent
```

```sh
uv run a2a/weather_agent
```

```sh
uv run a2a/supervisor
```

実行すると、Supervisorエージェントが起動し、各サブエージェントの`/.well-known/agent-card.json`にアクセスし、AgentCardを取得下のち、ユーザーからの質問に基づいて適切なサブエージェントにタスクを割り当て、結果を返すログが見れます。

### まとめ

本記事では、LangGraphを用いたマルチエージェントシステムの構築方法について解説しました。
特に、A2Aプロトコルを利用したエージェント間通信の実装に焦点を当てました。
Supervisorエージェントを中心に、各サブエージェントがどのように連携し、ユーザーのリクエストに応じて適切な情報を提供するかを示しました。

今後、LangGraphとA2Aプロトコルを組み合わせた疎結合なマルチエージェントシステムの開発をされる方の参考になれば幸いです。

### 参考にしたコード

#### Supervisor型マルチエージェント

https://github.com/langchain-ai/langgraph-supervisor-py

#### Adapter

https://github.com/a2aproject/a2a-samples/blob/main/samples/python/agents/langgraph/app/agent.py

#### AgentExecutor

https://github.com/a2aproject/a2a-samples/blob/main/samples/python/agents/langgraph/app/agent_executor.py

#### A2A Client

https://strandsagents.com/latest/documentation/docs/user-guide/concepts/multi-agent/agent-to-agent/#strands-a2a-tool

https://github.com/strands-agents/tools/blob/main/src/strands_tools/a2a_client.py
