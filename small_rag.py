import pymupdf as pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import create_retriever_tool
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain.messages import HumanMessage
from dotenv import load_dotenv
import fitz
import os

load_dotenv()
netherlands_facts = [
    "About 26percent of the Netherlands lies below sea level, protected by dikes and the Delta Works flood defense system.",
    "The country spans 41,850 square kilometers with over 18 million people, Europe's most densely populated nation.",
    "It leads in agricultural exports despite limited land, thanks to advanced farming techniques.",
    "The Port of Rotterdam is Europe's busiest port, and Schiphol Airport ranks fourth globally.",
    "First country to legalize same-sex marriage in 2001, with liberal policies on euthanasia and soft drugs.",
    "Dutch people are the world's tallest on average and own more bicycles than residents.",
    "A parliamentary constitutional monarchy since 1848, with Amsterdam as capital and The Hague as government seat.",
    "Features flat terrain, extensive canals, and borders the North Sea, Belgium, and Germany.",
    "Land reclamation via polders has added thousands of square kilometers since the 14th century.",
    "Divided into 12 provinces, 342 municipalities, and ancient water boards from 1196."
]

def load(text):

    doc=[]

    for page in text:
        text_part=page.strip()

        if text_part:
            doc.append(text_part)
    return doc
    


def split(doc):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=80,
        separators=["\n\n","\n","."," ",""],
        keep_separator=True,
        add_start_index=True,
        strip_whitespace=True
    )

    chunks=text_splitter.split_text("\n\n".join(doc))

    return chunks
    

def embed():
    embedding=embeddings=HuggingFaceEmbeddings(model='sentence-transformers/all-MiniLM-L6-v2')
    return embedding


def store(embedding,chunks):
    if not os.path.exists("text_rag.faiss"):
        vector_store=FAISS.from_texts(embedding=embedding,texts=chunks)
        vector_store.save_local("text.rag.faiss")
        return vector_store
    else:
        vector_store=FAISS.load_local("text_rag.faiss",allow_dangerous_deserialization=True,embeddings=embedding)
        return vector_store

def retreiver_tool_creation(vector_store):
    retreiver=vector_store.as_retriever(
        search_kwargs={
            "k":4
        }
    )

    retreiver_tool=create_retriever_tool(
        name="rag_search",
        description="""
    Searches the candidate's complete resume.
    Use this tool for EVERY question that requires ANY information from the resume.
    Always search first - never rely on your memory.
    """,
    retriever=retreiver
    )

    return retreiver_tool

def agent_creation(retriever_tool):
    model=ChatGroq(model='moonshotai/kimi-k2-instruct-0905')
    agent=create_agent(
        model=model,
        tools=[retriever_tool],
        system_prompt="""
        You are a precise fact analysis assistant.
     Your only source of truth is the users facts on netherlands.
     
     Rules you MUST follow:
     1. ALWAYS use the rag_search tool FIRST
     2. Base EVERY answer strictly on retrieved content
     3. If information is not found → say clearly "Information not found in facts"
     4. Be concise and factual
        """
    )
    return agent

def main():
    text = load(netherlands_facts)
    chunks = split(text)  # Fix chunk_size=450 in split()
    embedding = embed()
    vector_store = store(embedding=embedding, chunks=chunks)  # Fix args
    retriever_tool = retreiver_tool_creation(vector_store)
    agent = agent_creation(retriever_tool)  # Fix tools= not middleware
    
    query = "what is the population of netherlands?"
    res = agent.invoke({"messages": [HumanMessage(query)]})
    print(res["messages"][-1].content)

if __name__ == "__main__":
    main()