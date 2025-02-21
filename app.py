import os
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import streamlit as st

os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")
groq_api_key=os.getenv('GROQ_API_KEY')

model=ChatGroq(groq_api_key=groq_api_key,model_name="deepseek-r1-distill-llama-70b")

prompt=ChatPromptTemplate.from_template(
  """
  Answer the question based on the provided context only.
  Please provide the most accurate response based on the question
  <context>
  {context}
  <context>
  Question:{input}
"""
)

def create_vec_embedding():
  if "vectors" not in st.session_state:
    st.session_state.embeddings=OpenAIEmbeddings()
    st.session_state.loader=PyPDFDirectoryLoader("research_papers") #Data Ingestion Step
    st.session_state.docs=st.session_state.loader.load() #Document loading
    st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
    st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

st.title("RAG Document Q&A")
question=st.text_input("Enter your question regarding the research paper")

if st.button("Document Embedding"):
  create_vec_embedding()
  st.write("Vector Database is ready")

import time

if question:
  docs_chain=create_stuff_documents_chain(model,prompt)
  retriever=st.session_state.vectors.as_retriever()
  retrieval_chain=create_retrieval_chain(retriever,docs_chain)
  start_time=time.process_time()
  response=retrieval_chain.invoke({"input":question})
  print(f"Response Time:{time.process_time()-start_time}")
  st.write(response['answer'])
  # with a streamlit expander

  with st.expander("Document Similarity Search"):
    for i,doc in enumerate(response['context']):
      st.write(doc.page_content)
      st.write('--------------------------')

