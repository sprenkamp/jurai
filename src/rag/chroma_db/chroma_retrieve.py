from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
import chromadb

def retrieve_chromadb(collection_name: str="jurai"):
    persistent_client = chromadb.PersistentClient(path=f"data/chromadb/{collection_name}/")
    vectordb = Chroma(
        collection_name=collection_name,
        client=persistent_client,
        embedding_function=OpenAIEmbeddings(),
        )
    return vectordb
