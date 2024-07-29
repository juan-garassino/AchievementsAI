import os
import chromadb
import streamlit as st
from PIL import Image

from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
#from llama_index.llms.ollama import Ollama
from llama_index.llms.anthropic import Anthropic

from llama_index.core.node_parser import SentenceSplitter
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
#from langchain_community.chat_message_histories import ChatMessageHistory
from llama_index.core.prompts import PromptTemplate

# Configuration
CHROMA_DB_PATH = "./chroma_db/database"
COLLECTION_NAME = "achievementsAI"
ROOT= os.path.dirname(os.path.abspath(__file__))
DOCUMENTS_PATH = os.path.join(ROOT, 'data')
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "claude-3-sonnet-20240229" #"phi3"
LLM_TIMEOUT = 660.0
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
TOP_K = 2

# CUSTOM_PROMPT_TEMPLATE = (
#     "You are a friendly and helpful assistant designed to provide information about Juan's professional background. "
#     "You can mention just background in architecture but mainly focus on his last experiences in data science, machine learning, deep learning or generative AI engineering"
#     "You always talk about Juan in third person, you are not Juan Garassino, you are an assistant that he developed"
#     "Always respond in a positive and encouraging manner, using normal vocabulary. "
#     "Keep your answers short, concise, and straight to the point. "
#     "If you're unsure about something, it's okay to say you don't know. If you do not know is ok "
#     "You are an assisstant to help me get a job in science, machine learning, deep learning or generative AI engineering, never say I am better for architecture"
#     "Always mention last any skill from architecture, machine learning, generative AI are more impactfull"
#     "try to mention my architectural background as less as possible, focus on machine learning and generative AI"
#     "Here is the context information: {context_str}\n"
#     "Human: {query_str}\n"
#     "Assistant: ")

# Custom prompt template
CUSTOM_PROMPT_TEMPLATE = (
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
Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)

# # Get the Ollama base URL from environment variable
# ollama_base_url = os.getenv('OLLAMA_BASE_URL')

# # Initialize Ollama
# if ollama_base_url:
#     # If OLLAMA_BASE_URL is set, use it
#     Settings.llm = Ollama(model=LLM_MODEL, base_url=ollama_base_url, request_timeout=LLM_TIMEOUT)
# else:
#     # If OLLAMA_BASE_URL is not set, use default settings
#     Settings.llm = Ollama(model=LLM_MODEL, request_timeout=LLM_TIMEOUT)

ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

print(ANTHROPIC_API_KEY)

llm = Anthropic(
    model=LLM_MODEL,
    temperature=0.7,  # Adjust as needed
    api_key=ANTHROPIC_API_KEY,
    timeout=LLM_TIMEOUT,
    #streaming=True
)

# Create index
index = VectorStoreIndex.from_documents(
    documents, 
    storage_context=storage_context, 
    transformations=[SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)]
)

# Create query engine with custom prompt
custom_prompt = PromptTemplate(CUSTOM_PROMPT_TEMPLATE)
query_engine = index.as_query_engine(similarity_top_k=TOP_K,text_qa_template=custom_prompt)

# Get the PORT environment variable
port = int(os.environ.get('PORT', 8080))

# Configure Streamlit to use the specified port
st.set_option('server.port', port)
st.set_option('server.address', '0.0.0.0')

# Sidebar content
st.sidebar.title("About This App")

st.sidebar.markdown("""
This application is a chatbot designed to interact with a Retrieval-Augmented Generation (RAG) model. 
The RAG model has been fine-tuned with information about me and my professional curriculum.
""")

# Add an image to the sidebar

image_path = os.path.join(ROOT, 'assets', 'robot-assistant.png')

image = Image.open(image_path)  # Replace with the path to your image
st.sidebar.image(image, use_column_width=True)

st.sidebar.markdown("### 1. Key Features")
st.sidebar.markdown("""
- **Chatbot**: Engage with a sophisticated chatbot that provides information about me.
- **RAG Model**: Utilizing state-of-the-art Retrieval-Augmented Generation techniques for accurate responses.
""")

st.sidebar.markdown("### 2. Biography")
st.sidebar.markdown("""
Juan is a seasoned professional with a diverse background in architecture and deep learning. Holding a Master's degree in Architecture and Urbanism, and a Data Science & Data Engineering Certification, Juan has seamlessly transitioned from architecture to deep learning and MLOps engineering.
""")

st.sidebar.markdown("### 3. Skills and Expertise")
st.sidebar.markdown("""
- **Machine Learning & Deep Learning**
- **Computer Vision**
- **Generative AI**
- **Automated ML Pipelines**
- **3D Generative Architecture**
- **Smart Cities Urbanism**
""")

st.sidebar.markdown("### 4. Project Links")
st.sidebar.markdown("""
- **[Project 1](https://link-to-project1.com)**
- **[Project 2](https://link-to-project2.com)**
- **[Portfolio](https://link-to-portfolio.com)**
""")

st.sidebar.markdown("### 5. Testimonials")
st.sidebar.markdown("""
- "Juan is an outstanding professional who seamlessly blends creativity and technical expertise." - Colleague
- "His work in integrating AI with urbanism is truly pioneering." - Client
""")

st.sidebar.markdown("### 6. Download Resume")

CV = os.path.join(ROOT, 'assets', 'JuanGarassinoCV-2024.pdf')

#CV = '/Users/juan-garassino/Code/juan-garassino/mySandbox/testRag/LlamaChain/JuanGarassinoCV-2024.pdf'

# Custom CSS to make the button full width and use theme colors
st.markdown("""
<style>
div[data-testid="stDownloadButton"] > button {
    width: 100%;
    padding: 0.5rem;
    background-color: var(--primary-color);
    color: var(--text-color);
    border: 1px solid var(--primary-color);
    border-radius: 4px;
    cursor: pointer;
    transition: background-color 0.3s, color 0.3s, border-color 0.3s;
}
div[data-testid="stDownloadButton"] > button:hover {
    background-color: var(--background-color);
    color: var(--primary-color);
    border-color: var(--primary-color);
}
div[data-testid="stDownloadButton"] {
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# Use the sidebar for the download button
with st.sidebar:
    with open(CV, "rb") as file:
        st.download_button(
            label="Download PDF",
            data=file,
            file_name="JuanGarassinoCV-2024.pdf",
            mime="application/pdf",
            key="download_button"
        )

st.sidebar.markdown("### 7. Contact Me")

# Create a row of emojis with links
col1, col2, col3, col4, col5 = st.sidebar.columns(5)

with col1:
    st.markdown("[👨‍💻](https://github.com/juan-garassino)")

with col2:
    st.markdown("[💼](https://www.linkedin.com/in/juan-garassino/)")

with col3:
    st.markdown("[📷](https://www.instagram.com/artista.artificial/)")

with col4:
    st.markdown("[📧](mailto:juan.garassino@gmail.com)")

with col5:
    st.markdown("📱")  # Phone emoji without link

# Add phone number below the emojis
st.sidebar.markdown("<div style='text-align: center;'>+49 0152 24024860</div>", unsafe_allow_html=True)

# Custom CSS for button styling
st.markdown("""
<style>
.stButton > button {
    width: 100%;
    height: auto;
    white-space: normal;
    word-wrap: break-word;
    text-align: center;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 10px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# Streamlit UI
st.title("Chat with My Curriculum 🤓")
st.write("Welcome! Ask me anything about Juan's professional background. Use the chat or select any of the suggested questions!")

# Initialize chat history
history = StreamlitChatMessageHistory(key="chat_messages")

# Initialize session state
if 'button_pressed' not in st.session_state:
    st.session_state.button_pressed = False
if 'last_question' not in st.session_state:
    st.session_state.last_question = ""

# Function to process query and update chat
def process_query(query):
    st.chat_message("user").write(query)
    history.add_user_message(query)
    
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        response_placeholder.write("🧠 Thinking...")
        
        response = query_engine.query(query)
        
        response_placeholder.empty()
        response_placeholder.write(response.response)
        
        history.add_ai_message(response.response)

# Display chat history
for msg in history.messages:
    st.chat_message(msg.type).write(msg.content)

# If there's no chat history and no button has been pressed, show suggested questions
if not history.messages and not st.session_state.button_pressed:
    #st.markdown("#### Suggested Questions:")
    suggested_questions = [
        "Please tell me about Juan Garassino's skills in machine learning",
        "What can you tell me about Juan's early years?",
        "What do Juan's mom and dad do for a living?",
        "What was Juan's job while je lived in New Zealand?",
        "Would Juan be a good engineer for our Generative AI team?"
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