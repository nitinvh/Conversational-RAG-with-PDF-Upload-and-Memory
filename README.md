# Conversational RAG with PDF Upload and Memory
This project presents a practical and interactive application for document analysis, built with Streamlit. It implements a Retrieval-Augmented Generation (RAG) system with a PDF uploader and conversational memory, allowing users to upload a PDF file and engage in a context-aware conversation about its content.

## Project Description
Many businesses and individuals need to quickly find information and get insights from large documents without manual searching. This application solves that problem by transforming a PDF into a searchable knowledge base. The integrated conversational memory allows users to ask follow-up questions, making the interaction feel natural and intuitive, much like having a smart assistant who has just read your document.

The system is designed to:

Process PDF Documents: Accept and process PDF files uploaded by the user.

Maintain Conversational Context: Remember previous turns in a conversation to answer follow-up questions coherently.

Provide Grounded Answers: Generate accurate responses based solely on the content of the uploaded PDF, reducing LLM hallucinations.

Leverage High-Performance Models: Use a fast LLM and powerful embeddings for efficient analysis.

Offer a User-Friendly Interface: Provide a simple and intuitive Streamlit interface for seamless interaction.

## Features
PDF Document Upload: A Streamlit file uploader allows users to submit their own PDF files.

Document Chunking: Splits large PDF documents into manageable chunks for efficient embedding and retrieval.

Vectorization: Converts text chunks into numerical embeddings using Google Embeddings.

In-Memory Vector Store: Stores vectorized documents in a temporary, in-memory vector store (e.g., ChromaDB) for fast similarity search.

Conversational Memory: Uses LangChain's create_history_aware_retriever to intelligently rephrase queries based on chat history.

LLM-Powered Generation: Uses the powerful and fast Groq LLM to generate responses.

Session Management: Manages chat history within Streamlit's session_state to ensure continuity across user interactions.

## Technologies Used
Python: The core programming language.

Streamlit: For building the interactive and user-friendly web application interface.

LangChain: The framework that orchestrates the RAG pipeline, including document loading, text splitting, and retrieval.

PyPDF2 or PyMuPDF: Libraries used implicitly or explicitly to read and parse text from PDF files.

Google Embeddings: For generating high-quality vector representations of text.

Groq LLM: The fast language model that powers both the history-aware query generation and the final response synthesis.

ChromaDB (or other in-memory store): A lightweight vector database used for storing and searching document embeddings.

python-dotenv: For securely managing API keys and environment variables.

## Architecture and Workflow
The application's workflow can be seen in two main stages: a one-time document processing stage and a continuous chat stage.

Phase 1: Document Processing
File Upload: A user uploads a PDF file via the Streamlit interface.

PDF Reading: The content of the PDF is read and extracted as raw text.

Text Splitting: The large text document is broken down into smaller, semantically coherent chunks.

Vectorization & Storage: These text chunks are converted into numerical embeddings using Google Embeddings and stored in an in-memory vector store (like ChromaDB). This creates the searchable knowledge base for the RAG system.

Phase 2: Conversational Chat
User Input: The user types a question into the chat interface.

History-Aware Retrieval: The system uses the user's current question and the stored chat history to create a new, refined "standalone query." This process is handled by a history_aware_retriever.

Contextual Search: The standalone query is used to search the in-memory vector store, retrieving the most relevant text chunks from the uploaded PDF.

LLM Augmentation & Generation: The retrieved text chunks, along with the chat history and the user's original question, are sent to the Groq LLM. The LLM synthesizes this information to generate a comprehensive, grounded, and context-aware answer.

History Update: The new user question and the LLM's response are added to the st.session_state chat history, ensuring the conversation can continue seamlessly.

