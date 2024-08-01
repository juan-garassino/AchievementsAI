import os
import sys
import streamlit as st
from PIL import Image

from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext,
)
from llama_index.vector_stores import ChromaVectorStore
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.llms import Anthropic
from llama_index.node_parser import SentenceSplitter
from llama_index.prompts import PromptTemplate
from llama_index.memory import ChatMemoryBuffer

import chromadb

# Check if running on Streamlit Cloud
if os.getenv("MY_APP_ENV") == "streamlit_cloud":
    print("Running on Streamlit Cloud")
    # Replace sqlite3 with pysqlite3
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
else:
    print("Not running on Streamlit Cloud")

# Configuration
CHROMA_DB_PATH = "./chroma_db/database"
COLLECTION_NAME = "achievementsAI"
ROOT = os.path.dirname(os.path.abspath(__file__))
DOCUMENTS_PATH = os.path.join(ROOT, "data")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "claude-3-sonnet-20240229"
LLM_TIMEOUT = 660.0
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
TOP_K = 2

# Custom prompt template
CUSTOM_PROMPT_TEMPLATE = PromptTemplate(
    "You are a helpful assistant designed to provide information about Juan Garassino's professional background. "
    "Focus primarily on Juan's recent experiences in data science, machine learning, deep learning, and generative AI engineering. "
    "Always refer to Juan in the third person. You are an assistant he developed, not Juan himself. "
    "Respond positively and encouragingly, using clear and accessible language. "
    "Keep answers concise and directly address the query. "
    "If uncertain about any information, acknowledge that you don't know. "
    "Your goal is to assist Juan in securing positions in data science, machine learning, deep learning, or generative AI engineering. "
    "Emphasize Juan's skills and experiences in machine learning and generative AI as they are most relevant to his current career goals. "
    "If mentioning Juan's architectural background, do so briefly and mainly to highlight transferable skills or unique perspectives it brings to his current field. "
    "Context: {context_str}\n"
    "Human: {query_str}\n"
    "Assistant: "
)

# Set up Chroma database and vector store
db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Load documents
documents = SimpleDirectoryReader(DOCUMENTS_PATH).load_data()

# Configure settings
embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
llm = Anthropic(model=LLM_MODEL, temperature=0.7, api_key=ANTHROPIC_API_KEY, timeout=LLM_TIMEOUT)

node_parser = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model,
    node_parser=node_parser,
)

# Create index
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    service_context=service_context,
)

# Create query engine with custom prompt
query_engine = index.as_query_engine(
    similarity_top_k=TOP_K,
    text_qa_template=CUSTOM_PROMPT_TEMPLATE,
)

# Initialize chat memory
memory = ChatMemoryBuffer.from_defaults(token_limit=1500)

# Streamlit UI
st.title("Chat with Juan's Curriculum 🤓")
st.write(
    "Welcome to my AI-powered chatbot! I designed this app to answer questions about my professional background. Feel free to use the chat to interact with my curriculum. If you have further questions after the chat, let's organize a meeting - I'll be happy to answer the rest of them!"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to process query and update chat
def process_query(query):
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").write(query)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        response_placeholder.write("🧠 Thinking...")

        response = query_engine.query(query)
        memory.put(query, str(response))

        response_placeholder.empty()
        response_placeholder.write(response.response)

        st.session_state.messages.append({"role": "assistant", "content": response.response})

# Chat input
if prompt := st.chat_input("What would you like to know about Juan?"):
    process_query(prompt)

# (The sidebar content remains the same as in the original code)