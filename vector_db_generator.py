# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 12:54:42 2025

@author: Abhijit Mishra
"""

import os
import fitz
import pickle
from tqdm import tqdm
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

folder_name = 'arxiv_papers_astro_ph'

all_docs = []

for idx, filename in enumerate(os.listdir(folder_name)):
    if not filename.endswith('.pdf'):
        continue
        
    path = os.path.join(folder_name, filename)
    full_text = ''
    try:
        with fitz.open(path) as doc:
            for page in doc:
                full_text += page.get_text()
        
        doc_with_metadata = Document(
            page_content=full_text,
            metadata={"source": filename} 
        )
        all_docs.append(doc_with_metadata)
        
        if idx%10 == 0:
            print(f'{idx}/{len(os.listdir(folder_name))}')
        
    except Exception as e:
        print(f"Error processing {filename}: {e}")
    
    
        
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""] 
)

split_chunks = text_splitter.split_documents(all_docs)
print(f"Total chunks created: {len(split_chunks)}")

chunk_dict = {}

for chunk in split_chunks:
    source_file = chunk.metadata.get("source", "")
    
    paper_id = source_file.replace('.pdf', '')
    
    if paper_id not in chunk_dict:
        chunk_dict[paper_id] = []
    
    chunk_dict[paper_id].append(chunk.page_content)

with open('chunk_dict.pkl', 'wb') as f:
    pickle.dump(chunk_dict, f)


persist_directory = 'vector_db'

model_name = "BAAI/bge-base-en-v1.5"
model_kwargs = {'device': 'mps'} 
encode_kwargs = {'normalize_embeddings': True} 


embedding_function = HuggingFaceEmbeddings(model_name=model_name,model_kwargs=model_kwargs,encode_kwargs=encode_kwargs)


os.makedirs(persist_directory, exist_ok=True)

vectorstore = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_function
)

batch_size = 256

for i in tqdm(range(0, len(split_chunks), batch_size), desc="Ingesting chunks"):
    batch = split_chunks[i : i + batch_size]
    vectorstore.add_documents(documents=batch)

print("Vector store is ready.")


results = vectorstore.similarity_search("bullet galaxy cluster")

print(results[0].page_content)






'''
import pandas as pd

df = pd.read_parquet("hf://datasets/UniverseTBD/arxiv-qa-astro-ph/data/train-00000-of-00001-5a3764b3cbfd1977.parquet")

PROGRESS_FILE = "arxiv_results_with_progress (1).parquet"
df_progress = pd.read_parquet(PROGRESS_FILE)
df = df_progress.dropna(subset=['entry_id']) 


filtered_df = df[df['entry_id'].str.startswith("http://arxiv")]

filtered_df.to_csv('arxiv_df.csv', index=False)
'''





