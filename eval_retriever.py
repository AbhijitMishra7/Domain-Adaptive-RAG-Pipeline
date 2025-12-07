#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  7 04:06:53 2025

@author: abhijitmishra
"""

import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
import numpy as np

BASE_MODEL_ID = "BAAI/bge-base-en-v1.5"
FINETUNED_MODEL_PATH = "finetuned_astro_retriever"
DATASET_PATH = "golden_triplets_hybrid.csv"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
TOP_K = 10  

def get_rank(query_emb, relevant_doc_emb, candidate_embs):
    """
    Returns the rank of the relevant document among the candidates.
    Rank 1 means it was the top result.
    """
    all_candidates = np.vstack([relevant_doc_emb, candidate_embs])
    
    scores = np.dot(all_candidates, query_emb)
    
    sorted_indices = np.argsort(-scores) # Negative for descending sort
    
    rank = np.where(sorted_indices == 0)[0][0] + 1
    return rank

def evaluate_model(model_path, df, name="Model"):
    print(f"\n--- Evaluating {name} ---")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading {name} on {device}...")
    
    model = SentenceTransformer(model_path, device=device)
    
    ranks = []
    
    print("Computing metrics...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        query = row['query'] # Already has instruction from your CSV or add it here
        if not query.startswith("Represent"):
            query = QUERY_INSTRUCTION + query
            
        positive = row['positive']
        negative = row['negative']
        
        q_emb = model.encode(query, convert_to_numpy=True)
        p_emb = model.encode(positive, convert_to_numpy=True)
        n_emb = model.encode(negative, convert_to_numpy=True)
        
        sim_pos = np.dot(q_emb, p_emb)
        sim_neg = np.dot(q_emb, n_emb)
        
        if sim_pos > sim_neg:
            ranks.append(1) 
        else:
            ranks.append(2) 
            
    accuracy = sum([1 for r in ranks if r == 1]) / len(ranks)
    print(f"{name} Triplet Accuracy: {accuracy:.2%}")
    return accuracy

def full_retrieval_eval(model_path, df, chunk_dict_path, name="Model"):
    import pickle
    import numpy as np
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer(model_path, device=device)
    
    print(f"Loading full corpus from {chunk_dict_path}...")
    with open(chunk_dict_path, 'rb') as f:
        full_chunk_dict = pickle.load(f)
    
    corpus_docs = []
    for paper_id, chunks in full_chunk_dict.items():
        corpus_docs.extend(chunks)
        
    print(f"Encoding 'Haystack' of {len(corpus_docs)} chunks...")
    corpus_embs = model.encode(corpus_docs, show_progress_bar=True, convert_to_numpy=True)
    
    recalls = {1: 0, 5: 0, 10: 0}
    total_queries = len(df)
    
    print(f"Running Retrieval Evaluation for {name} on {total_queries} queries...")
    
    for idx, row in tqdm(df.iterrows(), total=total_queries):
        query = QUERY_INSTRUCTION + row['query'] if not row['query'].startswith("Represent") else row['query']
        target_passage = row['positive']
        
        try:
            target_idx = corpus_docs.index(target_passage)
        except ValueError:
            total_queries -= 1
            continue
            
        q_emb = model.encode(query, convert_to_numpy=True)
        
        scores = np.dot(corpus_embs, q_emb)
        
        top_k_indices = np.argsort(-scores)[:10]
        
        if target_idx in top_k_indices[:1]: recalls[1] += 1
        if target_idx in top_k_indices[:5]: recalls[5] += 1
        if target_idx in top_k_indices[:10]: recalls[10] += 1
        
    print(f"\nResults for {name}:")
    print(f"Recall@1:  {recalls[1]/total_queries:.2%}")
    print(f"Recall@5:  {recalls[5]/total_queries:.2%}")
    print(f"Recall@10: {recalls[10]/total_queries:.2%}")

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    df = pd.read_csv(DATASET_PATH)
    _, val_df = train_test_split(df, test_size=0.1, random_state=42)
    
    full_retrieval_eval(BASE_MODEL_ID, val_df, "chunk_dict.pkl", name="Base BGE")
    full_retrieval_eval(FINETUNED_MODEL_PATH, val_df, "chunk_dict.pkl", name="Finetuned Astro")
    





























