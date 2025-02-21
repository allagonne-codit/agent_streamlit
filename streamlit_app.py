import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA
#from agent import new_prompt
from huggingface_hub import InferenceClient
import os
# color palette
primary_color = "#1E90FF"
secondary_color = "#FF6347"
background_color = "#F5F5F5"
text_color = "#4561e9"

# Custom CSS
st.markdown(f"""
    <style>
    .stApp {{
        background-color: {background_color};
        color: {text_color};
    }}
    .stButton>button {{
        background-color: {primary_color};
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
    }}
    .stTextInput>div>div>input {{
        border: 2px solid {primary_color};
        border-radius: 5px;
        padding: 10px;
        font-size: 16px;
    }}
    .stFileUploader>div>div>div>button {{
        background-color: {secondary_color};
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
    }}
    </style>
""", unsafe_allow_html=True)

st.title("Basic Chatbot with Hugging Face")

HF_TOKEN = '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' #os.getenv("HF_TOKEN")
client = InferenceClient(
    model=None,#"meta-llama/Llama-2-7b-chat-hf"
    token=HF_TOKEN)
print("Model name:", client.model)
user_input = st.text_input("Your message:", value="")
if st.button("Send") and user_input:
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": user_input}],
        stream=False,
        max_tokens=50
    )
    st.write("Model name:", response.model, '\n')
    st.write("Answer:", response.choices[0].message.content)
    
