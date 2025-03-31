# System libraries
import os

# GenAI
import chromadb
from langchain.document_loaders import PyPDFDirectoryLoader 
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from sentence_transformers import SentenceTransformer

# const
CHROMA_PATH = "../chroma"
DATA_PATH = "../data"

def main():
    """
    Loads documents, split it into chunks, generate embeddings and creates the VectorDB
    """
    # Create the data store.
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)


def load_documents():
    """
    Load pdf document from ../data
    """
    loader = PyPDFDirectoryLoader(DATA_PATH)
    return loader.load()

def split_documents(documents: list[Document]):
    """
    Splits the document into chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 80,
        length_function =len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def get_embedding_function():
    """
    Creating embedding with all-MiniLM-L6-v2
    """
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return model.encode

def add_to_chroma(chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    vectorstore.add_documents(chunks)
    vectorstore.persist()

def add_to_chroma(chunks: list[Document]):
    """
    Initialize ChromaDB client, prepares the chunks and add them to the VectorDB
    """
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name="cookbook")

    # Preparing to be added in ChromaDB
    documents = []
    metadata = []
    ids = []

    i = 0

    for chunk in chunks:
        documents.append(chunk.page_content)
        ids.append("ID"+str(i))
        metadata.append(chunk.metadata)
        i+=1

    collection.upsert(
        documents=documents,
        metadatas = metadata,
        ids=ids
    )
    

if __name__ == "__main__":
    main()