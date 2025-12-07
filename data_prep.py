# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 11:01:30 2025

@author: Abhijit Mishra
"""

import arxiv
import pandas as pd
import os
import signal

def handler(signum, frame):
    raise TimeoutError("Row exceeded 5 minutes")

signal.signal(signal.SIGALRM, handler)

df = pd.read_parquet("hf://datasets/UniverseTBD/arxiv-qa-astro-ph/data/train-00000-of-00001-5a3764b3cbfd1977.parquet")

PROGRESS_FILE = "arxiv_results_with_progress (1).parquet"

if os.path.exists(PROGRESS_FILE):
    print(f"Loading progress from {PROGRESS_FILE}...")
    df = pd.read_parquet(PROGRESS_FILE)
else:
    print("Starting a new run...")
    df['title'] = pd.NA
    df['entry_id'] = pd.NA

client = arxiv.Client()

SAVE_EVERY_N_ROWS = 10 

rows_to_process = df[df['title'].isnull()].index
print(f"Total rows in DataFrame: {len(df)}")
print(f"Rows remaining to process: {len(rows_to_process)}")


for i, index in enumerate(rows_to_process):
    row = df.loc[index]
    print(i)
    try:
        signal.alarm(300) 
        
        query_text = row["answer"]
        search = arxiv.Search(
            query=f'all:{query_text} AND cat:astro-ph',
            max_results=1
        )

        result = next(client.results(search))
        
        signal.alarm(0)

        if result:
            df.loc[index, 'title'] = result.title
            df.loc[index, 'entry_id'] = result.entry_id
        else:
            df.loc[index, 'title'] = "NO_RESULT_FOUND"
            df.loc[index, 'entry_id'] = "NO_RESULT_FOUND"

    except Exception as e:
        print(f"\nError on row {index}: {e}")
        df.loc[index, 'title'] = "ERROR"
        df.loc[index, 'entry_id'] = str(e)

    if (i + 1) % SAVE_EVERY_N_ROWS == 0:
        df.to_parquet(PROGRESS_FILE)

print("\nProcessing complete. Saving final file.")
df.to_parquet(PROGRESS_FILE)
print("All done.")
