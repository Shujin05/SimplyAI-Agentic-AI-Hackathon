import os
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.retrievers import ParentDocumentRetriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, AgentType
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore
from langchain.tools import Tool

_agent = None 

# Load all documents from directory
def load_documents(directory_path: str):
    """Load all documents from a directory."""
    documents = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            loader = UnstructuredFileLoader(file_path, mode="single")
            docs = loader.load()
            documents.extend(docs)
    return documents

#   Initialise HuggingFace embeddings
def get_embeddings(model_name="sentence-transformers/all-mpnet-base-v2", device="cpu"):
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": False},
    )

#  Build or load FAISS vectorstore for child documents
def build_or_load_vectorstore(directory_path, hf_embedding, vectorstore_path="vectorstore.index", chunk_size=400):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size)
    
    if os.path.exists(vectorstore_path):
        # Load existing vectorstore
        vectorstore = FAISS.load_local(vectorstore_path, hf_embedding, allow_dangerous_deserialization=True)
    else:
        # Load documents only if vectorstore doesn't exist
        documents = load_documents(directory_path)
        child_docs = splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(child_docs, hf_embedding)
        vectorstore.save_local(vectorstore_path)
    
    return vectorstore, splitter

#  Create or load parent document store
def build_or_load_docstore(docstore_path="parent_docstore", chunk_size=2000):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size)

    # Ensure the folder exists
    os.makedirs(docstore_path, exist_ok=True)

    fs = LocalFileStore(docstore_path)
    store = create_kv_docstore(fs)

    return store, splitter

# Create ParentDocumentRetriever 
def create_retriever(vectorstore, docstore, child_splitter, parent_splitter, k=4, score_threshold=0.5):
    return ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_type="similarity",
        search_kwargs={"score_threshold": score_threshold, "k": k},
    )

#  Initialise ChatOpenAI model
def create_llm(api_key: str, model="gpt-4o-mini", temperature=0):
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        openai_api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )

#  Create a RetrievalQA chain
def create_qa_chain(llm, retriever):
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
    )

def make_rag_tool(qa_chain):
    def rag_tool_func(query: str):
        result = qa_chain.invoke(query)
        return {"output": str(result)} 
    return rag_tool_func
    
# Initialise the agent with RAG Tool.
def create_agent(qa_chain, llm):
    tools = [
        Tool(
            name="RAG Tool",
            func=make_rag_tool(qa_chain),
            description="Answers user's legal-related queries based on MOM's website"
        )
    ]
    
    return initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )


def initialize_agentic_ai(directory_path: str, api_key: str,
                          vectorstore_path="vectorstore.index",
                          docstore_path="parent_docstore"):    
    hf_embedding = get_embeddings()
    
    vectorstore, child_splitter = build_or_load_vectorstore(
        directory_path, hf_embedding, vectorstore_path
    )
    docstore, parent_splitter = build_or_load_docstore(docstore_path)
    
    retriever = create_retriever(vectorstore, docstore, child_splitter, parent_splitter)
    llm = create_llm(api_key)
    qa_chain = create_qa_chain(llm, retriever)
    agent = create_agent(qa_chain, llm)
    
    return agent

def get_agent():
    global _agent
    if _agent is None:
        API_KEY = ""
        DIRECTORY_PATH = "C:/Users/User/Downloads/SimplyAI-Agentic-AI-Hackathon/backend/data_ingress/mom.gov.sg/webpage"
        _agent = initialize_agentic_ai(directory_path=DIRECTORY_PATH, api_key=API_KEY)
    return _agent
