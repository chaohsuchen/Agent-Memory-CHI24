
"""
chroma 版本 save db
說明: 
langchain + Chromadb
用於讀取指定目錄中的文本文件，將其切割並轉換為向量表示，然後創建並持久化一個文本資料庫
A utility for reading text files from a specified directory, segmenting and converting them into vector representations, and then creating and persisting a text database.
"""

import os
import shutil
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import Chroma
from langchain_chroma import Chroma
from typing import List
from langchain.docstore.document import Document

from langchain_community.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)

# Map file extensions to document loaders and their arguments
loader_mapping = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    # Add more mappings for other file extensions and loaders as needed
}

load_dotenv(".\..\config\.env")
api_key = os.getenv("OPENAI_API_KEY")



def load_single_document(file_path: str) -> List[Document]:
    """
    載入單一文件
    根據文件路徑和文件類型，使用對應的加載器加載文檔。

    Args:
        file_path (str): 要加載的文件路徑。

    Returns:
        List[Document]: 加載後的文檔列表。
    """
    # 根據檔案格式選擇相對應的加載器
    ext = "." + file_path.rsplit(".", 1)[-1].lower()
    if ext in loader_mapping:
        loader_class, loader_args = loader_mapping[ext]
        loader = loader_class(file_path, **loader_args)

        document = loader.load()
        for doc in document:
            source = doc.metadata["source"].split('\\')[-1].replace(".txt", "")
            doc.metadata["source"] = source

        return document

    raise ValueError(f"Unsupported file extension '{ext}'")


def load_documents(directory_path) -> List[Document]:
    """
    載入目錄中的所有文檔
    遍歷指定目錄，對每個文件使用 load_single_document 函數進行加載。

    Args:
        directory_path (str): 要遍歷加載的目錄路徑。

    Returns:
        List[Document]: 目錄中所有文檔的列表。
    """
    file_paths = [os.path.join(directory_path, file) for file in os.listdir(
        directory_path)]
    documents = []
    for file_path in file_paths:
        print(file_path)
        documents.extend(load_single_document(file_path))
    return documents


def generate_db(directory_path: str, persist_directory: str) -> None:
    """
    切割文本、embedding、創建db
    根據指定的目錄和持久化目錄，切割文本並創建 Chroma 資料庫。
    切割方式可選擇 CharacterTextSplitter() or RecursiveCharacterTextSplitter()

    Args:
        directory_path (str): 要讀取的目錄路徑。
        persist_directory (str): 持久化目錄的路徑。
    """

    # text_splitter = CharacterTextSplitter(separator="\n", chunk_size=800, chunk_overlap=100)
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    # Verify the API key is set in OpenAIEmbeddings
    print(f"Using API key: {api_key[:4]}...")  # Print only the first few characters of the API key for security


    documents_with_metadata = load_documents(directory_path)
    docs = text_splitter.split_documents(documents_with_metadata)

    # 為了避免 rate limit error，批次處理文檔
    total_length = len(docs)
    batch_size = 32

    for batch_start in range(0, total_length, batch_size):
        # Verify the API key is set in OpenAIEmbeddings
        print(f"Using API key: {api_key[:4]}...")  # Print only the first few characters of the API key for security
        batch_end = min(batch_start + batch_size, total_length)
        batch_texts = docs[batch_start:batch_end]
        db = Chroma.from_documents(
            documents=batch_texts, embedding=embeddings, persist_directory=persist_directory)
        print(f"Inserted {batch_end}/{total_length} chunks")

    result = db.similarity_search("Why Injection Molding?")
    print(result)

def delete_chroma_db(persist_directory: str):
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
        print(f"Deleted existing ChromaDB at {persist_directory}")

if __name__ == "__main__":
    try:
        persist_directory = ".\\test_db"
        delete_chroma_db(persist_directory)
        generate_db("C:\\Users\\user\\Desktop\\LLM_understands\\Agent-Memory-CHI24\\rag\\data", persist_directory)
        print("Done!")
    except Exception as e:
        print(f"An error occurred: {e}")
    
