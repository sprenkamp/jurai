import argparse
import sys
sys.path.append("src/langchain_agent/helper/")
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import chromadb
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from src.rag.chroma_db.helper.split import create_documents_from_texts

def process_multiple_rows(rows, collection_name):
    metadata_list = []
    texts = []
    for row in rows:
        metadata = {'user': row["User"]}
        metadata_list.append(metadata)
        text = row['Assistant']
        texts.append(text)

    docs = create_documents_from_texts(texts, metadata_list)
    Chroma.from_documents(
        collection_name=collection_name,
        client=chromadb.PersistentClient(path=f"data/chromadb/{collection_name}/"),
        documents=docs, 
        embedding=OpenAIEmbeddings(),
    )
    
def save_data(collection_name):
    df = pd.read_csv("data/train.csv")
    # remove rows with none string values
    print(df.shape)
    df= df[df[['System', 'User', 'Assistant']].apply(lambda row: all(isinstance(cell, str) for cell in row), axis=1)]
    print(df.shape)
    # Create batches of rows
    batch_size = 1000  
    rows = [row for _, row in df.iterrows()]
    row_batches = [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]
    for batch in tqdm(row_batches):
        process_multiple_rows(batch, collection_name)

if __name__ == "__main__":  
    # run the method
    collection_name = "jurai"
    save_data(collection_name)
    