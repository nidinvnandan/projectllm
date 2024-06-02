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

# Initialize retriever
rerank_retriever = vector()

# Output parser
output_parser = StrOutputParser()

# Prompt templates
instruction_to_system = """
Given a chat history and the latest user question 
which might reference context in the chat history, formulate a standalone question 
which can be understood without the chat history. Do NOT answer the question, 
just reformulate it if needed and otherwise return it as is.
"""

question_maker_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", instruction_to_system),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

question_chain = question_maker_prompt | llm | StrOutputParser()

qa_system_prompt = """be act like a HR officer and answer the questions to the employee
{context}
Question: {question}
Helpful Answer:"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

# Function to generate contextualized question
def contextualized_question(input: dict):
    if input.get("chat_history"):
        return question_chain.invoke(input)
    else:
        return input["question"]

# Set up retrieval chain
async def retriever_chain(input):
    context = contextualized_question(input)
    return await rerank_retriever.arun({"query": context})

# Final RAG chain
async def rag_chain(input):
    context = await retriever_chain(input)
    qa_prompt_input = {
        "context": context,
        "question": input["question"]
    }
    return await llm.arun(qa_prompt_input)

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Function to reset chat history
def reset_chat_history():
    st.session_state.chat_history = []

# New Chat button
if st.button('New Chat'):
    reset_chat_history()

# Custom CSS for chat messages
st.markdown("""
    <style>
    .bot-message {
        text-align: left;
        background-color: #f1f1f1;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        color: black.
    }
    .human-message {
        text-align: right;
        background-color: #e1f5fe.
        padding: 10px.
        border-radius: 5px.
        margin: 10px 0.
        color: black.
    }
    </style>
""", unsafe_allow_html=True)

# Display chat history
for i in range(0, len(st.session_state.chat_history), 2):
    human_message = st.session_state.chat_history[i].content
    ai_message = st.session_state.chat_history[i+1].content if i+1 < len(st.session_state.chat_history) else ''
    st.markdown(f"<div class='human-message'><strong>You:</strong> {human_message}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='bot-message'><strong>Bot:</strong> {ai_message}</div>", unsafe_allow_html=True)

# Input query
query = st.text_input('Enter the query')

# Function to answer the question
async def answer_question(question):
    recent_history = st.session_state.chat_history[-14:] if len(st.session_state.chat_history) > 14 else st.session_state.chat_history
    ai_msg = await rag_chain({"question": question, "chat_history": recent_history})
    st.session_state.chat_history.extend([HumanMessage(content=question), ai_msg])
    if len(st.session_state.chat_history) > 14:
        st.session_state.chat_history = st.session_state.chat_history[-14:]
    return ai_msg.content

# Answer the query on button press
if st.button('➤'):
    if query:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(answer_question(query))
        st.markdown(result)
    else:
        st.write("Please enter a query.")
