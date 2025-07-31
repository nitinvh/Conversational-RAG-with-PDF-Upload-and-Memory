import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts  import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
import streamlit as st



load_dotenv()

os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

st.title("Conversational RAG With PDF uplaods and chat history")
st.write("Upload Pdf's and chat with their content")

import asyncio # <-- Add this import

# ... other imports ...

# Use caching to create the embeddings instance safely.
# Manually run the async creation within a synchronous context.
@st.cache_resource
def get_embeddings():
    # This inner function is now an async function
    async def _create_embeddings_client():
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Use asyncio.run() to execute the async function and get the result
    return asyncio.run(_create_embeddings_client())

# Use caching for the LLM as well. This is typically synchronous.
@st.cache_resource
def get_llm():
    return ChatGroq(model="llama-3.3-70b-versatile")

# Now call the cached functions to get your instances
embeddings = get_embeddings()
llm = get_llm()

uploaded_file = st.file_uploader("Choose a PDF file", type='pdf', accept_multiple_files=True)
session_id=st.text_input("Session ID",value="default_session")
    
if 'store' not in st.session_state:
        st.session_state.store={}
        
if uploaded_file:
    documents=[]
    for uploaded_f in uploaded_file:
        tempPdf = f"./temp.pdf"
        with open(tempPdf,'wb') as file:
            file.write(uploaded_f.getvalue())
            file_name = uploaded_f.name
        
        loader = PyPDFLoader(tempPdf)
        docs = loader.load()
        documents.extend(docs)
    
    # Split docs and store it in vectorDB
    splitter = RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=700)
    splitted_docs = splitter.split_documents(documents)
    vectorStore = Chroma.from_documents(documents=splitted_docs,embedding=embeddings)
    retriever = vectorStore.as_retriever()

    contextualize_q_system_prompt=(
            "Given a chat history and the latest user question"
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
    
    history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)
    # Answer question
    system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use three sentences maximum and keep the "
                "answer concise."
                "\n\n"
                "{context}"
            )
    qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
        
    question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
    rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

    def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]
        
    conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
    user_input = st.text_input("Your question:")
    if user_input:
        session_history=get_session_history(session_id)
        response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable": {"session_id":session_id}
                },  # constructs a key "abc123" in `store`.
            )
        st.write(st.session_state.store)
        st.write("Assistant:", response['answer'])
        st.write("Chat History:", session_history.messages)


        

