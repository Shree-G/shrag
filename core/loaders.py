# The purpose of this file is to load all the necessary data sources in and format them into documents

from langchain_core.documents import Document
from langchain.document_loaders import PyPDFLoader

def load_pdf_documents(file_path: str) -> list[Document]:
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

