import streamlit as st
import os
import json
import tiktoken
import pandas as pd
import torch

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

st.set_page_config(page_title="Flykite Airlines HR Assistant", layout="wide")

@st.cache_resource
def load_embedding_model():
    return SentenceTransformerEmbeddings(model_name='thenlper/gte-large')

@st.cache_resource
def load_vector_store(embedding_model):
    pdf_path = 'data/Dataset - Flykite Airlines_ HRP.pdf' # Path within the Docker container
    out_dir = 'vector_db'

    if not os.path.exists(out_dir):
        # If vector_db doesn't exist, create it from the PDF
        pdf_loader = PyMuPDFLoader(pdf_path)
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name='cl100k_base',
            chunk_size=512,
            chunk_overlap= 20
        )
        document_chunks = pdf_loader.load_and_split(text_splitter)
        vectordb = Chroma.from_documents(
            documents=document_chunks,
            embedding=embedding_model,
            persist_directory=out_dir
        )
        vectordb.persist() # Ensure persistence after creation

    # Load the vector store (it will load if it exists, or use the newly created one)
    return Chroma(persist_directory=out_dir, embedding_function=embedding_model)

@st.cache_resource
def load_llm():
    model_name_or_path = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
    model_basename = "mistral-7b-instruct-v0.2.Q6_K.gguf"
    model_path = hf_hub_download(
        repo_id=model_name_or_path,
        filename=model_basename
    )
    return Llama(
        model_path=model_path,
        n_ctx=5000,
        n_gpu_layers=-1 if torch.cuda.is_available() else 0, # Use all GPU layers if available
        n_batch=512
    )

# Load models and vector store
embedding_model = load_embedding_model()
vectorstore = load_vector_store(embedding_model)
llm = load_llm()

# Retriever
retriever = vectorstore.as_retriever(
    search_type='similarity',
    search_kwargs={'k': 3} # Using k=3 as it performed well in tuning
)

# QnA System Message (from notebook)
qna_system_message = """
You are an empathetic and knowledgeable HR Assistant for Flykite Airlines. Your task is to provide clear, concise, and helpful answers to employee questions based on the provided context.
The context will be provided with source references (e.g., "[Page X]").
Please use the provided context to answer the user's question. If you cannot find a direct answer, try to infer based on the information available and mention any limitations. Always cite the page numbers from the source document for each piece of information you provide.
If the context does not contain enough information to fully answer the question, state that you cannot fully answer the question based on the provided documents and suggest contacting HR.
Do not generate information that is not supported by the context.
"""

qna_user_message_template = """
###Context
Here are some documents that are relevant to the question mentioned below.
{context}

###User Role
{user_role}

###Question
{question}
"""

def generate_rag_response(user_input, user_role="General Employee", k=3, max_tokens=200, temperature=0.1, top_p=0.9, top_k=40):
    # Retrieve relevant document chunks
    relevant_document_chunks = retriever.get_relevant_documents(query=user_input)

    # Combine document chunks with page numbers into a single context
    context_with_sources = []
    for i, d in enumerate(relevant_document_chunks):
        context_with_sources.append(f"{d.page_content} [Page {d.metadata['page'] + 1}]")
    context_for_query = "\n\n".join(context_with_sources)

    user_message = qna_user_message_template.replace('{context}', context_for_query)
    user_message = user_message.replace('{user_role}', user_role)
    user_message = user_message.replace('{question}', user_input)

    prompt = qna_system_message + '\n' + user_message

    # Generate the response
    try:
        response = llm(
                  prompt=prompt,
                  max_tokens=max_tokens,
                  temperature=temperature,
                  top_p=top_p,
                  top_k=top_k
                  )

        # Extract and print the model's response
        response = response['choices'][0]['text'].strip()
    except Exception as e:
        response = f'Sorry, I encountered the following error: \n {e}'

    return response

# Streamlit UI
st.title("\ud83d\udeeb Flykite Airlines HR Assistant")
st.write("Hello! I am your AI HR Assistant for Flykite Airlines. Ask me anything about our HR policies!")

user_question = st.text_input("Your Question:", "What are the effects on the benefits I receive if my probation is extended?")
user_role = st.selectbox("Your Role:", ["General Employee", "Pilot", "Manager", "HR Staff"], index=0)

if st.button("Get Answer"): # Updated to a button for explicit action
    with st.spinner("Fetching your answer..."):
        rag_answer = generate_rag_response(user_question, user_role=user_role)
        st.markdown(f"**HR Assistant:** {rag_answer}")

st.markdown("---\n_Disclaimer: This chatbot provides information based on the Flykite Airlines HR Policy Handbook. For specific cases or official decisions, please consult with your HR department directly._")
