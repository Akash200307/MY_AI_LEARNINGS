import pymupdf as pdf
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import create_retriever_tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os
import fitz
import time



load_dotenv()
doc =fitz.open("resume.pdf")

text=[]
for page in doc:
    text_part = page.get_text("text")
    
    if text_part.strip():           # skip completely empty pages
        text.append(text_part)
doc.close()

text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=450,
    chunk_overlap=80,
    separators=["\n\n", "\n", ".", " ", ""],
    keep_separator=True,
    add_start_index=True,
    strip_whitespace=True   
    
)

chunks=text_splitter.split_text("/n/n".join(text))


embeddings=HuggingFaceEmbeddings(model='sentence-transformers/all-MiniLM-L6-v2')
model=ChatGroq(model='moonshotai/kimi-k2-instruct-0905')


if not os.path.exists("resume_rag.faiss"):
    vector_store=FAISS.from_texts(texts=chunks,embedding=embeddings,metadatas=[{"source":"resume.pdf","chunk_index":i} for i in range(len(chunks))])
    vector_store.save_local("resume_rag.faiss")

else:
    vector_store=FAISS.load_local("resume_rag.faiss",embeddings=embeddings,allow_dangerous_deserialization=True)


retriever=vector_store.as_retriever(search_kwargs={
    'k':4
})

retriever_tool = create_retriever_tool(
    name="resume_search",
    description=(
        """Searches the candidate's complete resume.
    Use this tool for EVERY question that requires ANY information from the resume.
    Always search first - never rely on your memory."""
    ),
    retriever=retriever,
)





agent=create_agent(
    model=model,
    tools=[retriever_tool],
    system_prompt="""You are a precise resume analysis assistant.
     Your only source of truth is the candidate's resume.
     
     Rules you MUST follow:
     1. ALWAYS use the search_resume tool FIRST
     2. Base EVERY answer strictly on retrieved content
     3. If information is not found → say clearly "Information not found in resume"
     4. Be concise and factual
     5. Use bullet points when listing skills/experiences""",
)
st=time.perf_counter()
response=agent.invoke({"messages":[HumanMessage("what are the technical skills mentioned in the resume only on software?")]})
end=time.perf_counter()

print(response["messages"][-1].content)

excution_time=end-st

print(f"{excution_time:.3f}",seconds)
