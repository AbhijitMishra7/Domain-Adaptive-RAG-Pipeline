# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 15:40:24 2025

@author: Abhijit Mishra
"""

import os
import re
import pickle
import random
import pandas as pd
from tqdm import tqdm
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

REDUNDANCY_MARGIN = 0.15   # sScore difference required to be a negative
MIN_POSITIVE_SCORE = 0.5   # Minimum score for a chunk to be considered the true answer

CROSS_ENCODER_MODEL_ID = 'BAAI/bge-reranker-base' 
EMBEDDING_MODEL_ID = "BAAI/bge-base-en-v1.5"
VECTOR_DB_PATH = 'vector_db'


print("Initializing models...")

model = CrossEncoder(CROSS_ENCODER_MODEL_ID)

embedding_function = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_ID,
    model_kwargs={'device': 'mps'}, 
    encode_kwargs={'normalize_embeddings': True}
)


if os.path.exists(VECTOR_DB_PATH):
    print(f"Loading vector store from {VECTOR_DB_PATH}...")
    vectorstore = Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embedding_function
    )
else:
    raise ValueError(f"Vector DB not found at {VECTOR_DB_PATH}. Please run the ingestion script first.")

print("Loading dataset...")
df = pd.read_csv('arxiv_df.csv')

def extract_id_from_url(url):
    if isinstance(url, str):
        return url.split('/')[-1]
    return str(url)

df['entry_id'] = df['entry_id'].apply(extract_id_from_url)


with open('chunk_dict.pkl', 'rb') as f:
     raw_chunk_dict = pickle.load(f)

chunk_dict = {}

for filename_key, chunks in raw_chunk_dict.items():
    clean_key = filename_key.replace('.pdf', '')
    
    parts = clean_key.split('.')
    if len(parts) >= 2 and parts[0].isdigit():
        clean_id = f"{parts[0]}.{parts[1]}"
        chunk_dict[clean_id] = chunks
    
    elif 'v' in clean_key: 
         possible_id = clean_key.split('_')[0]
         chunk_dict[possible_id] = chunks
         
    chunk_dict[clean_key] = chunks

print(f"Loaded {len(chunk_dict)} papers into chunk dictionary.")


def is_junk(text):
    """Filters out low-quality chunks (bibliographies, tables, short text)."""
    # 1. Too short (titles, affiliations)
    if len(text.split()) < 40: return True
    
    # 2. Reference list detection
    text_lower = text.strip().lower()
    if text_lower.startswith("references") or text_lower.startswith("bibliography"):
        return True
    
    # 3. Number density (Data tables)
    digit_count = sum(c.isdigit() for c in text)
    if len(text) > 0 and (digit_count / len(text) > 0.20): return True
    
    return False



triplet_dataset = []
metrics_log = []

print("Starting Triplet Generation...")

for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    
    paper_id = row['entry_id']
    query = row['question']
    answer = row['answer']
    
    metric_entry = {
        'paper_id': paper_id,
        'query_length': len(query.split()),
        'num_chunks': 0,
        'best_pos_score': 0.0,
        'global_neg_found': False,
        'global_neg_score': 0.0,
        'local_neg_found': False,
        'local_neg_score': 0.0,
        'selected_neg_type': None,
        'outcome': 'skipped'
    }

    if paper_id not in chunk_dict:
        metric_entry['outcome'] = 'missing_paper_text'
        metrics_log.append(metric_entry)
        continue

    chunks_raw = chunk_dict[paper_id]
    valid_chunks = [c for c in chunks_raw if not is_junk(c)]
    metric_entry['num_chunks'] = len(valid_chunks)

    if not valid_chunks: 
        metric_entry['outcome'] = 'no_valid_chunks'
        metrics_log.append(metric_entry)
        continue 

    qc_pairs = [(query, chunk) for chunk in valid_chunks]
    ac_pairs = [(answer, chunk) for chunk in valid_chunks]
    
    try:
        qc_scores = model.predict(qc_pairs)
        ac_scores = model.predict(ac_pairs)
    except Exception as e:
        print(f"Error scoring {paper_id}: {e}")
        continue
    
    scored_chunks = []
    for i, chunk in enumerate(valid_chunks):
        final_score = (0.3 * qc_scores[i]) + (0.7 * ac_scores[i])
        scored_chunks.append({'text': chunk, 'score': final_score})
        
    scored_chunks.sort(key=lambda x: x['score'], reverse=True)
    
    best_match = scored_chunks[0]
    metric_entry['best_pos_score'] = float(best_match['score']) 
    
    if best_match['score'] < MIN_POSITIVE_SCORE:
        metric_entry['outcome'] = 'weak_positive'
        metrics_log.append(metric_entry)
        continue
        
    positive_passage = best_match['text']
    
    # Find Candidates (Deterministic Search)
    
    # A. Global Negative Search (Inter-Paper)
    global_neg_cand = None
    global_neg_score = 0.0
    
    try:
        results = vectorstore.similarity_search(query, k=10)
        
        external_candidates = [
            doc.page_content for doc in results 
            if doc.metadata.get('source', '').replace('.pdf', '') != paper_id
        ]
        
        if external_candidates:
            ext_pairs = [(query, cand) for cand in external_candidates]
            ext_scores = model.predict(ext_pairs)
            
            best_ext_idx = -1
            best_ext_score = -1.0
            
            for i, s in enumerate(ext_scores):
                if s < (best_match['score'] - REDUNDANCY_MARGIN):
                    if s > 0.05 and s > best_ext_score:
                        best_ext_score = s
                        best_ext_idx = i
            
            if best_ext_idx != -1:
                global_neg_cand = external_candidates[best_ext_idx]
                global_neg_score = float(best_ext_score)
                metric_entry['global_neg_found'] = True
                metric_entry['global_neg_score'] = global_neg_score
                
    except Exception as e:
        pass

    # B. Local Negative Search (Intra-Paper)
    local_neg_cand = None
    local_neg_score = 0.0
    
    score_threshold = best_match['score'] - REDUNDANCY_MARGIN
    
    for cand in scored_chunks[1:]:
        if cand['score'] < score_threshold:
            if cand['score'] > 0.05:
                local_neg_cand = cand['text']
                local_neg_score = float(cand['score'])
                metric_entry['local_neg_found'] = True
                metric_entry['local_neg_score'] = local_neg_score
                break
    
    # Selection Logic (Hardest Negative Mining)
    final_negative = None
    
    candidates = []
    
    if global_neg_cand:
        candidates.append({
            'text': global_neg_cand,
            'score': global_neg_score,
            'type': 'global'
        })
        
    if local_neg_cand:
        candidates.append({
            'text': local_neg_cand,
            'score': local_neg_score,
            'type': 'local'
        })
    
    if candidates:
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        best_choice = candidates[0]
        final_negative = best_choice['text']
        metric_entry['selected_neg_type'] = best_choice['type']
        metric_entry['outcome'] = 'success'
        
        triplet_dataset.append([query, positive_passage, final_negative])
    else:
        metric_entry['outcome'] = 'no_negative_found'

    metrics_log.append(metric_entry)


print(f"Done! Generated {len(triplet_dataset)} triplets.")

triplet_df = pd.DataFrame(triplet_dataset, columns=['query', 'positive', 'negative'])
triplet_df.to_csv("golden_triplets_hybrid.csv", index=False)
print("Saved triplets to golden_triplets_hybrid.csv")

metrics_df = pd.DataFrame(metrics_log)
metrics_df.to_csv("triplet_generation_metrics.csv", index=False)
print("Saved metrics to triplet_generation_metrics.csv")







































