import google.generativeai as genai
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import urllib
import warnings
from pathlib import Path as p
from pprint import pprint
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import pandas as pd
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_community.llms import Cohere
import streamlit as st
from IPython.display import display
from IPython.display import Markdown
import textwrap
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough


st.header("HR ChatBot")
os.environ["GOOGLE_API_KEY"] = 'AIzaSyA7u2G1y06P1KFfGuT1gVhj-8JbV409NdM'
os.environ["COHERE_API_KEY"] = 'ErFPP679z7c4wPQLv8zp63JRblE8N2I4N1LxGmUQ'
llm = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.7)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
@st.cache_resource
def load_and_split_pdf(pdf_path):
    loader = PyPDFLoader("ZETA_CORPORATION.pdf")

    docs = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    documents = text_splitter.split_documents(docs)
    return documents
pdf_path = "ZETA_CORPORATION.pdf"
documents = load_and_split_pdf(pdf_path)
