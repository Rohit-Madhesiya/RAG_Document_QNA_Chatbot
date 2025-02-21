import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain,create_history_aware_retriever
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["HUGGING_FACE_ACCESS_TOKEN"]=os.getenv("HUGGING_FACE_ACCESS_TOKEN")

embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# streamlit frontend

st.title("Conversational RAG with PDF uploads and Chat History")
st.write("Upload PDFs and chat with their content")

# input Groq API Key
api_key=st.text_input("Enter Groq API Key: ",type="password")

if api_key:
  model=ChatGroq(model_name="deepseek-r1-distill-llama-70b",groq_api_key=api_key)
  # chat interface

  session_id=st.text_input("Session ID",value="default_session")
  # statefully manage the chat history

  if 'store' not in st.session_state:
    st.session_state.store={}
  
  uploaded_files=st.file_uploader("Choose a PDF File",type="pdf",accept_multiple_files=False)
  
  # process the uploaded PDF
  if uploaded_files:
    
    documents=[]
    
    if os.path.exists("temp_pdf"):
      if os.path.isfile("temp_pdf"):
        os.remove("temp_pdf")
      else:
        import shutil
        shutil.rmtree("temp_pdf")

    os.makedirs("temp_pdf")

    pdf_dir=os.path.join("temp_pdf",uploaded_files.name)

    with open(pdf_dir,"wb") as f:
      f.write(uploaded_files.getbuffer())

    loader=PyPDFDirectoryLoader("temp_pdf")
    docs=loader.load()
    documents.extend(docs)
    
    # splits and creates embedding for the documents
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=500)
    splits=text_splitter.split_documents(documents)
    vector_store=Chroma.from_documents(documents=splits,
                                       embedding=embeddings,
                                       persist_directory="chroma_db"
                                       )
    retriever=vector_store.as_retriever()
    if os.path.exists(pdf_dir):
      os.remove(pdf_dir)
  
    contextualize_q_system_prompt=(
      "Given a chat history and the latest user question"
      "which might reference context in the chat history"
      "formulate a standalone question which can be understood"
      "without the chat history. Do NOT answer the question"
      "just reformulate it if needed and otherwirs return it as is."
    )
    
    contextualize_q_prompt=ChatPromptTemplate.from_messages([
      ("system",contextualize_q_system_prompt),
      MessagesPlaceholder("chat_history"),
      ("human","{input}")
    ])

    history_aware_retriever=create_history_aware_retriever(model,retriever,contextualize_q_prompt)

    # answer question prompt
    system_prompt=(
      "You are an assistant for question-answering tasks "
      "Use the followinf pieces of retrieved context to answer "
      "the question. if you don't know the answer, say that you "
      "don't know. Use three sentences maximum and keep the "
      "answer concise "
      "\n\n"
      "{context}"
    )

    qa_prompt=ChatPromptTemplate.from_messages([
      ("system",system_prompt),
      MessagesPlaceholder("chat_history"),
      ("human","{input}")
    ])

    question_answer_chain=create_stuff_documents_chain(model,qa_prompt)
    
    rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

    def get_session_history(session_id:str)->BaseChatMessageHistory:
      if session_id not in st.session_state.store:
        st.session_state.store[session_id]=ChatMessageHistory()
      return st.session_state.store[session_id]

    conversational_rag_chain=RunnableWithMessageHistory(
      rag_chain,get_session_history,
      input_messages_key="input",
      history_messages_key="chat_history",
      output_messages_key="answer"
    )

    user_input=st.text_input("Your question:")
    if user_input:
      session_history=get_session_history(session_id)
      response=conversational_rag_chain.invoke(
        {"input":user_input},
        config={
          "configurable":{"session_id":session_id}
        }
      )
      st.write(st.session_state.store)
      st.write("Assistant: ",response['answer'])
      st.write("Chat History:",session_history.messages)
else:
  st.warning("Please enter the Groq API Key")




