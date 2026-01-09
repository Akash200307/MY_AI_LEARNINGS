from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain.agents.middleware import ModelRequest,ModelResponse,wrap_model_call
load_dotenv()

high_model=ChatGroq(model="moonshotai/kimi-k2-instruct-0905")
low_model=ChatGroq(model="llama-3.3-70b-versatile")


@wrap_model_call
def dynamic_model(request:ModelRequest,handler)->ModelResponse:
    message_count=len(request.state['messages'])

    if message_count>2:
        model=high_model
    else:
        model=low_model
    request.model=model
    return handler(request)

agent= create_agent(
    model=low_model,
    middleware=[dynamic_model]

)
human=HumanMessage("what is pi?")
system=SystemMessage("You are a helpful assistant")

response=agent.invoke(
    {
        "messages":[
            system,human,HumanMessage("relation between pi and world"),
        ]
    }
)

print(response["messages"][-1].content)
print(response["messages"][-1].response_metadata['model_name'])