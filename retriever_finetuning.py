#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 09:00:41 2025

@author: abhijitmishra
"""

import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from sentence_transformers.trainer import SentenceTransformerTrainer
from torch.utils.data import DataLoader
import torch
import gc
import tqdm
from sklearn.model_selection import train_test_split

MODEL_ID = "BAAI/bge-base-en-v1.5"
DATASET_PATH = "golden_triplets_hybrid.csv"
OUTPUT_PATH = "finetuned_astro_retriever"
BATCH_SIZE = 8
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

tqdm.autonotebook.tqdm = tqdm.std.tqdm
tqdm.autonotebook.trange = tqdm.std.trange

gc.collect()
try:
    torch.mps.empty_cache()
except:
    pass

device = "mps"
print(f"Using device: {device}")

df = pd.read_csv(DATASET_PATH)
df['query'] = QUERY_INSTRUCTION + df['query']

train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

train_examples = [InputExample(texts=[row['query'], row['positive'], row['negative']]) for _, row in train_df.iterrows()]
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE, pin_memory=False)

model = SentenceTransformer(MODEL_ID, device=device)

#Gradient Checkpointing for preventing OOM
model[0].auto_model.gradient_checkpointing_enable()

train_loss = losses.MultipleNegativesRankingLoss(model=model)

val_evaluator = evaluation.TripletEvaluator(
    anchors=val_df['query'].tolist(),
    positives=val_df['positive'].tolist(),
    negatives=val_df['negative'].tolist(),
    name="astro_val_evaluator"
)

_original_compute_loss = SentenceTransformerTrainer.compute_loss
def _patched_compute_loss(self, model, inputs, return_outputs=False, **kwargs):
    return _original_compute_loss(self, model, inputs, return_outputs=return_outputs)
SentenceTransformerTrainer.compute_loss = _patched_compute_loss

print("Starting training...")
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=val_evaluator,
    epochs=NUM_EPOCHS,
    warmup_steps=int(len(train_dataloader) * NUM_EPOCHS * 0.1),
    optimizer_params={'lr': LEARNING_RATE},
    output_path=OUTPUT_PATH,
    show_progress_bar=True,
    save_best_model=True
)

print(f"Training complete. Model saved to {OUTPUT_PATH}")

finetuned_model = SentenceTransformer(OUTPUT_PATH, device=device)
test_query = QUERY_INSTRUCTION + "What is the velocity of the bullet cluster?"
emb = finetuned_model.encode(test_query)
print(f"Verification successful. Embedding shape: {emb.shape}")

















































