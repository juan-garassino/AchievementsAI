import os
import sys
import streamlit as st
from PIL import Image
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatAnthropic
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.sql_database import SQLDatabase
from langchain.chains import SQLDatabaseChain
import pinecone

# Check if running on Streamlit Cloud
if os.getenv("MY_APP_ENV") == "streamlit_cloud":
    print("Running on Streamlit Cloud")
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
else:
    print("Not running on Streamlit Cloud")

# Configuration
ROOT = os.path.dirname(os.path.abspath(__file__))
DOCUMENTS_PATH = os.path.join(ROOT, "data")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "claude-3-sonnet-20240229"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
TOP_K = 2

# Pinecone configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME = "juan-curriculum"

# PostgreSQL configuration
POSTGRES_URI = os.getenv("POSTGRES_URI")

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

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# Load documents
loader = DirectoryLoader(DOCUMENTS_PATH, glob="**/*.txt")
documents = loader.load()

# Split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
texts = text_splitter.split_documents(documents)

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Create or load Pinecone index
if PINECONE_INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(PINECONE_INDEX_NAME, dimension=384)  # Dimension for 'all-MiniLM-L6-v2'

# Create vector store
vectorstore = Pinecone.from_documents(
    documents=texts,
    embedding=embeddings,
    index_name=PINECONE_INDEX_NAME
)

# Initialize LLM
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
llm = ChatAnthropic(
    model=LLM_MODEL,
    temperature=0.7,
    anthropic_api_key=ANTHROPIC_API_KEY,
    max_tokens=1000
)

# Initialize SQL database
db = SQLDatabase.from_uri(POSTGRES_URI)
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

# Create tools for the agent
tools = [
    Tool(
        name="Curriculum Search",
        func=vectorstore.similarity_search,
        description="Useful for answering questions about Juan's professional background and experiences."
    ),
    Tool(
        name="Personal Database",
        func=db_chain.run,
        description="Useful for querying structured data about Juan, such as specific dates, numbers, or categorical information."
    )
]

# Create conversation memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize the agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
    prompt=CUSTOM_PROMPT_TEMPLATE
)

# Streamlit UI
st.title("Chat with Juan's AI Agent 🤖")
st.write(
    "Welcome to my AI-powered agent! This assistant can access both my curriculum vitae and a database with structured information about me. Feel free to ask questions about my professional background or specific details."
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

        response = agent.run(query)

        response_placeholder.empty()
        response_placeholder.write(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

# Chat input
if prompt := st.chat_input("What would you like to know about Juan?"):
    process_query(prompt)

# (The sidebar content remains the same as in the original code)