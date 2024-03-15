# Chatbot-Template


Source code temp:

from langchain.llms import Ollama
import streamlit as st
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chains import RetrievalQA

# Initialize Ollama
ollama = Ollama(base_url='http://localhost:11434', model="mistral-jp-local")

# Ask a question that llm doesn't know yet...
question = "what does further do?"
answer = ollama(question)
st.write("Question: what does further do?")
st.write("Answer:", answer)

# Input for URL
user_input_url = st.text_input("Enter a website URL", "https://gofurther.com")

if user_input_url:
    # Load website
    loader = WebBaseLoader(user_input_url)
    data = loader.load()

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)

    # Get embeddings
    oembed = OllamaEmbeddings(base_url="http://localhost:11434", model="mistral-jp-local")
    vectorstore = FAISS.from_documents(documents=all_splits, embedding=oembed)

question = st.text_input("Enter a question",question)

# Submit button
if st.button('Submit'):
    # Search for similar docs from vector store
    #docs = vectorstore.similarity_search(question)
    #st.write("Documents similar to the question:", docs)

    # Retrieve answer using llm and vector store
    qachain = RetrievalQA.from_chain_type(ollama, retriever=vectorstore.as_retriever())
    answer = qachain.invoke({"query": question})
    st.write("Answer:", answer)



https://learnlangchain.streamlit.app/



