Why this matters?

Traditional research tools like the ArXiv API rely on metadata and keyword matching, forcing researchers to download multiple PDFs and manually search for answers. This pipeline transforms that workflow.
By ingesting the full text of papers into a semantic vector store, we move beyond simple document retrieval to true Knowledge Extraction. This system allows users to ask natural language questions across the entire corpus and receive faithfully cited answers grounded in specific paragraphs, capturing semantic nuances that keyword searches miss completely.

The Data
This project is built upon the UniverseTBD/arxiv-qa-astro-ph dataset.

Source: Hugging Face: UniverseTBD/arxiv-qa-astro-ph

To link the isolated Q&A pairs to their original context, utilized a reverse-search heuristic by mapping answers to source papers via the ArXiv API. The pipeline executes a precise search query (all:{answer} AND cat:astro-ph) to locate the original document, effectively using the answer text as a search key to download the source PDF.

Models:
Embedding model: BAAI/bge-base-en-v1.5
Cross Encoder: BAAI/bge-reranker-base
QA LLM: meta-llama/Meta-Llama-3-8B

Embedding model finetuning using contrastive learning with MultipleNegativesRankingLoss

Vector DB: Chroma
