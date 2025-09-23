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