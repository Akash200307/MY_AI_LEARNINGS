import pymupdf as pdf
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import create_retriever_tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

import fitz

load_dotenv()
doc =fitz.open("resume.pdf")

text=[]
for page in doc:
    blocks=page.get_text("blocks")
    text.extend( block[4] for block in blocks if block[4].strip())

embeddings=HuggingFaceEmbeddings(model='sentence-transformers/all-MiniLM-L6-v2')
model=ChatGroq(model='moonshotai/kimi-k2-instruct-0905')

vector_store=FAISS.from_texts(texts=text,embedding=embeddings)
vector_store.save_local("resume.index")
retriever=vector_store.as_retriever(search_kwargs={
    'k':3
})

retriever_tool = create_retriever_tool(
    name="resume_search",
    description=(
        "Search the candidate's resume content. "
        "Use this tool EVERY time you need ANY information from the resume."
    ),
    retriever=retriever,
)

agent=create_agent(
    model=model,
    tools=[retriever_tool],
    system_prompt="You are a resume analysis assistant. Always use the 'resume_search' tool first "
        "Always use the 'resume_search' tool FIRST to retrieve relevant information before answering. "
        "Base your answers strictly on the retrieved context."
)

response=agent.invoke({"messages":[HumanMessage("what are the technical skills mentioned in the resume?")]})
print(response["messages"][-1].content)
