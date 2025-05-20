from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# Load documents
documents = SimpleDirectoryReader("./data").load_data()

# Initialize Chroma client and collection
db = chromadb.PersistentClient(path="./chroma_db")

graph_collection = db.get_or_create_collection("graph")

graph_vector_store = ChromaVectorStore(chroma_collection=graph_collection)

vector_collection = db.get_or_create_collection("vector")

vector_store = ChromaVectorStore(chroma_collection=vector_collection)

# Define the ingestion pipeline
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=512, chunk_overlap=20),
        HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5"),
    ],
    vector_store=vector_store,
)

# Run the pipeline on your documents
nodes = pipeline.run(documents=documents)


import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore



# Initialize Chroma client and collection



# Set up ChromaVectorStore and StorageContext
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Build and persist the index
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
index.storage_context.persist(persist_dir="./storage")


from llama_index.core.graph_stores import SimplePropertyGraphStore
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
client = chromadb.PersistentClient("./chroma_db")
collection = client.get_or_create_collection("my_graph_vector_db")


index = PropertyGraphIndex.from_documents(
    documents,
    embed_model=OpenAIEmbedding(model_name="text-embedding-3-small"),
    graph_store=SimplePropertyGraphStore(),
    vector_store=ChromaVectorStore(collection=collection),
    show_progress=True,
)
index.storage_context.persist(persist_dir="./storage")