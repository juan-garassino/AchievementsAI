import os
import sys
import chromadb
import streamlit as st
from PIL import Image
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.chat_models import ChatAnthropic
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import StreamlitChatMessageHistory
from langchain.prompts import PromptTemplate

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
    input_variables=["context", "question"],
    template="""
You are a helpful assistant designed to provide information about Juan Garassino's professional background. 
Focus primarily on Juan's recent experiences in data science, machine learning, deep learning, and generative AI engineering. 
Always refer to Juan in the third person. You are an assistant he developed, not Juan himself. 
Respond positively and encouragingly, using clear and accessible language. 
Keep answers concise and directly address the query. 
If uncertain about any information, acknowledge that you don't know. 
Your goal is to assist Juan in securing positions in data science, machine learning, deep learning, or generative AI engineering. 
Emphasize Juan's skills and experiences in machine learning and generative AI as they are most relevant to his current career goals. 
If mentioning Juan's architectural background, do so briefly and mainly to highlight transferable skills or unique perspectives it brings to his current field. 
Context: {context}
Human: {question}
Assistant: """
)

# Load documents
loader = DirectoryLoader(DOCUMENTS_PATH, glob="**/*.txt")
documents = loader.load()

# Split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
texts = text_splitter.split_documents(documents)

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Create vector store
vectorstore = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory=CHROMA_DB_PATH,
    collection_name=COLLECTION_NAME
)

# Initialize LLM
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
llm = ChatAnthropic(
    model=LLM_MODEL,
    temperature=0.7,
    anthropic_api_key=ANTHROPIC_API_KEY,
    max_tokens=1000
)

# Create conversation chain
conversation = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": TOP_K}),
    combine_docs_chain_kwargs={"prompt": CUSTOM_PROMPT_TEMPLATE}
)

# Streamlit UI
st.title("Chat with Juan's Curriculum 🤓")
st.write(
    "Welcome to my AI-powered chatbot! I designed this app to answer questions about my professional background. Feel free to use the chat to interact with my curriculum. If you have further questions after the chat, let's organize a meeting - I'll be happy to answer the rest of them!"
)

# Initialize chat history
history = StreamlitChatMessageHistory(key="chat_messages")

# Initialize session state
if "button_pressed" not in st.session_state:
    st.session_state.button_pressed = False
if "last_question" not in st.session_state:
    st.session_state.last_question = ""

# Function to process query and update chat
def process_query(query):
    st.chat_message("user").write(query)
    history.add_user_message(query)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        response_placeholder.write("🧠 Thinking...")

        response = conversation({"question": query, "chat_history": [(msg.type, msg.content) for msg in history.messages]})

        response_placeholder.empty()
        response_placeholder.write(response['answer'])

        history.add_ai_message(response['answer'])

# Display chat history
for msg in history.messages:
    st.chat_message(msg.type).write(msg.content)

# If there's no chat history and no button has been pressed, show suggested questions
if not history.messages and not st.session_state.button_pressed:
    suggested_questions = [
        "Can you describe Juan Garassino's skills in machine learning?",
        "What can you share about Juan's early years?",
        "How might Juan's experience as a lecturer benefit our company?",
        "What was Juan's role while he lived in New Zealand?",
        "Would Juan be a strong fit for our Generative AI team?",
    ]

    for question in suggested_questions:
        if st.button(question):
            st.session_state.button_pressed = True
            st.session_state.last_question = question
            st.rerun()

# If a button was just pressed, process the query
if st.session_state.button_pressed and not history.messages:
    process_query(st.session_state.last_question)
    st.session_state.button_pressed = False
    st.session_state.last_question = ""

# Chat input
if prompt := st.chat_input():
    process_query(prompt)

# (The sidebar content remains the same as in the original code)