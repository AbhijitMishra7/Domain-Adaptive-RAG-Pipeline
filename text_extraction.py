# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 12:31:13 2025

@author: Abhijit Mishra
"""

import arxiv
import pandas as pd
from tqdm.auto import tqdm
import os
import time

PROGRESS_FILE = "arxiv_results_with_progress.parquet"
DOWNLOAD_DIR = "arxiv_papers_astro_ph"

os.makedirs(DOWNLOAD_DIR, exist_ok=True)
print(f"Papers will be downloaded to: {DOWNLOAD_DIR}")

df = pd.read_parquet(PROGRESS_FILE)
df = df.dropna()
df = df[df['title'] != 'NO_RESULT_FOUND']

def extract_id_from_url(url):
    if not isinstance(url, str):
        return None
    try:
        return url.split('/')[-1]
    except Exception:
        return None

all_paper_ids = df['entry_id'].apply(extract_id_from_url).dropna().unique()

ids_to_download = []
existing_files = os.listdir(DOWNLOAD_DIR)

for paper_id in all_paper_ids:
    if not any(f.startswith(paper_id) for f in existing_files):
        ids_to_download.append(paper_id)

print(f"Already downloaded: {len(all_paper_ids) - len(ids_to_download)} papers.")
print(f"Remaining to download: {len(ids_to_download)} papers.")


client = arxiv.Client(delay_seconds = 3, num_retries = 5)

if not ids_to_download:
    print("All papers are already downloaded.")
else:
    print("Starting download process...")
    pbar = tqdm(ids_to_download, desc="Downloading PDFs")

    for paper_id in pbar:
        pbar.set_description(f"Downloading: {paper_id}")
        
        try:
            search_by_id = arxiv.Search(id_list=[paper_id], max_results=1)
            paper = next(client.results(search_by_id), None)
            if paper:
                paper.download_pdf(dirpath=DOWNLOAD_DIR)    
            else:
                print(f"\nWarning: Could not find paper with ID {paper_id}")

        except Exception as e:
            print(f"\nError downloading {paper_id}: {e}")
            time.sleep(10)
        
        break

print("\nDownload process complete.")











