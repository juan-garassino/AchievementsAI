import os
import hashlib
import json
from datetime import datetime
import chromadb
from llama_index import SimpleDirectoryReader, StorageContext
from llama_index.vector_stores import ChromaVectorStore

# Constants
CHROMA_DB_PATH = "path/to/your/chroma_db"
COLLECTION_NAME = "your_collection_name"
DOCUMENTS_PATH = "path/to/your/documents"
HASH_FILE_PATH = "file_hashes.json"

def setup_chroma_db():
    db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return storage_context

def load_documents():
    return SimpleDirectoryReader(DOCUMENTS_PATH).load_data()

def get_file_hash(filepath):
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def load_stored_hashes():
    if os.path.exists(HASH_FILE_PATH):
        with open(HASH_FILE_PATH, "r") as f:
            return json.load(f)
    return {}

def save_file_hashes(hashes):
    with open(HASH_FILE_PATH, "w") as f:
        json.dump(hashes, f)

def check_for_changes(directory):
    changes = {"new": [], "modified": [], "deleted": []}
    current_files = {}
    
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            current_files[filename] = get_file_hash(filepath)
    
    stored_hashes = load_stored_hashes()
    
    for filename, current_hash in current_files.items():
        if filename not in stored_hashes:
            changes["new"].append(filename)
        elif stored_hashes[filename] != current_hash:
            changes["modified"].append(filename)
    
    for filename in stored_hashes:
        if filename not in current_files:
            changes["deleted"].append(filename)
    
    return changes, current_files

def update_vector_database(storage_context, changes, current_files):
    for new_file in changes["new"]:
        document = SimpleDirectoryReader(os.path.join(DOCUMENTS_PATH, new_file)).load_data()[0]
        storage_context.add_documents([document])
        print(f"Added new file: {new_file}")
    
    for modified_file in changes["modified"]:
        # For simplicity, we'll remove and re-add modified documents
        # In a real-world scenario, you might want to update them in place
        storage_context.delete_ref_doc(modified_file)
        document = SimpleDirectoryReader(os.path.join(DOCUMENTS_PATH, modified_file)).load_data()[0]
        storage_context.add_documents([document])
        print(f"Updated modified file: {modified_file}")
    
    for deleted_file in changes["deleted"]:
        storage_context.delete_ref_doc(deleted_file)
        print(f"Removed deleted file: {deleted_file}")
    
    save_file_hashes(current_files)

def main():
    print("Setting up Chroma database...")
    storage_context = setup_chroma_db()
    
    print("Checking for changes in documents...")
    changes, current_files = check_for_changes(DOCUMENTS_PATH)
    
    if any(changes.values()):
        print("Changes detected. Updating vector database...")
        update_vector_database(storage_context, changes, current_files)
        print("Vector database updated successfully.")
    else:
        print("No changes detected. Vector database is up to date.")
    
    print("Loading all documents...")
    documents = load_documents()
    print(f"Loaded {len(documents)} documents.")

if __name__ == "__main__":
    main()