from dotenv import load_dotenv
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest,ModelResponse,dynamic_prompt
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
load_dotenv()

@dataclass
class Context:
    user_role:str

@dynamic_prompt
def user_role_prompt(request:ModelRequest)->str:
    user_role=request.runtime.context.user_role

    base_prompt="you are a helpful and concise assisstant"
    match user_role:
        case 'expert':
            return f'{base_prompt} Provide detail and technial responses'
        case 'child':
            return f'Explain like I am a five yet old'
        case _:
            return base_prompt

model=ChatGroq(model='moonshotai/kimi-k2-instruct-0905')

agent =create_agent(
    model=model,
    middleware=[user_role_prompt],
    context_schema=Context
)

message=HumanMessage("Explain pydantic")

response=agent.invoke(
    {
        'messages':[message]
    },
    context=Context(user_role="expert")

)


print(response["messages"][-1].content)
        

