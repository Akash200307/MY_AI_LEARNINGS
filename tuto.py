import requests
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool,ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver

from dataclasses import dataclass
load_dotenv()
@dataclass
class Context:
    user_id:str

@dataclass
class ResponseFormat:
    summary:str
    celsius:float
    fahrenheit:float
    humdity:float

@tool('get_weather',description="Return weather information for a given city",return_direct=False)
def get_weather(city:str):
    response=requests.get(f'http://wttr.in/{city}?format=j1')
    return response.json()

@tool('fetch_location',description="locate users city")
def locate_user(runtime: ToolRuntime[Context]):
    match runtime.context.user_id:
        case 'A123':
            return 'chennai'
        case 'B123':
            return 'madurai'
        case 'C123':
            return 'pondicherry'
        case _:
            return 'UNKNOWN'
        
checkpointer=InMemorySaver()

            

agent=create_agent(
    model=ChatGoogleGenerativeAI(model="gemini-3-flash-preview"),
    tools=[get_weather,locate_user],
    system_prompt="you are a weather assisstant ,who always give a fact about the area.",
    context_schema=Context,
    response_format=ResponseFormat
)

config={
    'configurable':
    {
        'thread_id':1
     }
}

response=agent.invoke({
    'messages':[
        {'role':'user',
         'content':'What is the weather'
         }
    ]},
    config=config,
    context=Context(user_id='123')
    )


# print(response['messages'][-1].content)
print(response['structured_response'])
print(response['structured_response'].summary)