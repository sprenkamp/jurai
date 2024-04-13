from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
import json
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base", chunk_size=4094, chunk_overlap=512
)

file_path = 'data/train_datasmall.json'
with open(file_path, 'r') as file:
    lines = file.readlines()
    old_list = [json.loads(line) for line in lines]

new_list = []

for doc in tqdm(old_list):
    if len(enc.encode(doc['messages'][2]["content"])) > 4096:
        chunks = text_splitter.split_text(doc['messages'][2]["content"])
        for chunk in chunks:
            new_doc = doc.copy()
            new_doc['messages'][2]["content"] = chunk
            new_list.append(doc)
    else:
        new_list.append(doc)

print(len(old_list), len(new_list))

with open('data/train_datasmall_chunked_4069.jsonl', 'w') as file:
    for doc in new_list:
        file.write(json.dumps(doc) + '\n')