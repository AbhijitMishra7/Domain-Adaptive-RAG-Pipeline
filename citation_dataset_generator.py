#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 31 14:20:00 2025

@author: Abhijit Mishra
"""

import os
import json
import argparse
from typing import List, Dict

import pandas as pd
from tqdm import tqdm

# Optional: use OpenAI but fallback to local model if unavailable
try:
    import openai  # type: ignore
except ImportError:
    openai = None

PROMPT_FAITHFUL = (
    "You are a helpful research assistant. Your task is to answer the question using *only* the provided context.\n"
    "- Cite your source using the format [Paper {paper_id}].\n"
    "- Do not use any outside knowledge.\n"
    "- If the context does not contain the answer, state that clearly.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
)

PROMPT_REFUSAL = (
    "You are a helpful research assistant. Look at the provided context and question.\n"
    "Your task is to generate a new, related *follow-up* question that CANNOT be answered by the context.\n\n"
    "Context:\n{context}\n\n"
    "Original Question: {question}\n"
)

def call_teacher(prompt: str, model: str = "gpt-4o", temperature: float = 0.2) -> str:
    """Calls the teacher model (OpenAI) or raises informative error."""
    if openai is None:
        raise ImportError("openai package not installed. Install or replace teacher model.")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    openai.api_key = api_key

    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message["content"].strip()


def build_examples(row: Dict[str, str], include_refusal: bool = True) -> List[Dict[str, str]]:
    """Generates faithful (and optionally refusal) examples from a single golden triplet row."""
    q = row["question"]
    ctx = row["positive_passage"]
    paper_id = row["paper_id"]
    faithful_prompt = PROMPT_FAITHFUL.format(context=ctx, question=q, paper_id=paper_id)
    faithful_answer = call_teacher(faithful_prompt)

    examples = [
        {
            "context": ctx,
            "question": q,
            "answer": faithful_answer,
            "paper_id": paper_id,
            "label": "faithful",
        }
    ]

    if include_refusal:
        refusal_prompt = PROMPT_REFUSAL.format(context=ctx, question=q)
        hallucination_q = call_teacher(refusal_prompt)
        refusal_answer = (
            "I am sorry, but the provided context does not contain information to answer that question."
        )
        examples.append(
            {
                "context": ctx,
                "question": hallucination_q,
                "answer": refusal_answer,
                "paper_id": paper_id,
                "label": "refusal",
            }
        )
    return examples


def main():
    parser = argparse.ArgumentParser(description="Generate citation-aligned dataset using teacher model.")
    parser.add_argument("--golden_csv", required=True, help="Path to golden triplet CSV file.")
    parser.add_argument("--output", required=True, help="Output JSONL file path.")
    parser.add_argument("--max_rows", type=int, default=None, help="Optional cap on rows processed.")
    parser.add_argument("--skip_existing", action="store_true", help="Skip rows already in output file.")
    args = parser.parse_args()

    # Load existing examples if any (for resuming)
    existing_questions = set()
    if args.skip_existing and os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    existing_questions.add(data["question"])
                except Exception:
                    continue
        print(f"Loaded {len(existing_questions)} existing examples to skip.")

    df = pd.read_csv(args.golden_csv)
    if args.max_rows:
        df = df.head(args.max_rows)

    with open(args.output, "a", encoding="utf-8") as writer:
        for _, row in tqdm(df.iterrows(), total=len(df)):
            if row["question"] in existing_questions:
                continue
            try:
                examples = build_examples(row)
                for ex in examples:
                    writer.write(json.dumps(ex, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"Skipping row due to error: {e}")


if __name__ == "__main__":
    main()

