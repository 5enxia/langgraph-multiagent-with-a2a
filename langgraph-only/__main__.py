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
