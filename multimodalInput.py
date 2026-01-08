from dotenv import load_dotenv
from base64 import b64encode
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()

chatModel=ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

message={
    'role':'user',
    'content':[
        {'type':'text','text':'Describe the image'},
        {
            'type':'image',
            'base64':b64encode(open('sample.png','rb').read()).decode(),
            'mime_type':'image/png'
            }
    ]

}

response=chatModel.invoke([message])
print(response.content)