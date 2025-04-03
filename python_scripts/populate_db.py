# System libraries
import os
import openai

# GenAI
from langchain.document_loaders import PyPDFLoader 
from langchain_openai import OpenAIEmbeddings
from langchain.schema.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from dotenv import load_dotenv

# const
CHROMA_PATH = "../chroma"
DATA_PATH = "../data/Student-Cookbook.pdf"

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
    loader = PyPDFLoader(DATA_PATH)
    return loader.load()

def split_documents(documents: list[Document]):
    """
    Splits the document into chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function =len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]):
    """
    Initialize ChromaDB client, prepares the chunks and add them to the VectorDB
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY") 
    openai.api_key = api_key

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    return Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)


if __name__ == "__main__":
    main()