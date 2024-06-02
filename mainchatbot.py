import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_cohere import CohereRerank
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import asyncio

# Streamlit header
st.header("HR ChatBot")

# Environment setup
os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY')
os.environ["COHERE_API_KEY"] = os.getenv('COHERE_API_KEY')

# Initialize language model and embeddings
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Function to load and split PDF
@st.cache_resource
def load_and_split_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_documents(docs)

# Load documents from PDF
pdf_path = "ZETA_CORPORATION.pdf"
documents = load_and_split_pdf(pdf_path)

# Function to set up retrievers
@st.cache_resource
def vector():
    vector = FAISS.from_documents(documents, embeddings)
    retriever = vector.as_retriever(search_kwargs={"k": 10})
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever, search_kwargs={"k": 8}
    )
    rerank_retriever = ContextualCompressionRetriever(
        base_compressor=CohereRerank(top_n=5), base_retriever=compression_retriever
    )
    return rerank_retriever


