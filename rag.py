from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import create_retriever_tool
load_dotenv()


embeddings=HuggingFaceEmbeddings(model='sentence-transformers/all-MiniLM-L6-v2')
model=ChatGroq(model='moonshotai/kimi-k2-instruct-0905')
texts = [
    'Apple makes very good computers.',
    'I believe Apple is innovative!',
    'I love apples.',
    'I am a fan of MacBooks.',
    'I enjoy oranges.',
    'I like Lenovo Thinkpads.',
    'I think pears taste very good.'
]

vector_store=FAISS.from_texts(texts=texts,embedding=embeddings)
retriever=vector_store.as_retriever(search_kwargs={'k':3})

retriever_tool=create_retriever_tool(retriever=retriever,name='kb_search',description="search the small product /fruit knowledge base for information")

agent=create_agent(
    model=model,
    tools=[retriever_tool],
    system_prompt=(
    "You are a helpful assistant answering questions about computers, laptops, Apple products, and fruits mentioned in the knowledge base. "
        "Always use the 'kb_search' tool FIRST to retrieve relevant information before answering. "
        "Base your answers strictly on the retrieved context."
    )
)

response=agent.invoke({"messages":[HumanMessage("what are the fruits liked by the person and disliked by the person")]})
print(response["messages"][-1].content)

# print(vector_store.similarity_search('I love food ',k=7))