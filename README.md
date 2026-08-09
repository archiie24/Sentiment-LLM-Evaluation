# LLM Evaluation Framework

A Treater-inspired framework for evaluating customer email sentiment using a local LLM (**Ollama + Llama 3.2**). The project compares multiple prompting strategies, uses an LLM judge to resolve disagreements, applies confidence-based review rules, and supports **human-in-the-loop validation** for reliable sentiment classification.

## Features

- Dual-prompt sentiment evaluation
- Batch inference with Ollama (Llama 3.2)
- LLM-as-a-Judge comparison within the same LLM call
- Confidence-based uncertainty detection
- Human review queue generation
- Persistent LLM results for reproducible human review
- Final approved sentiment export
- Gold dataset generation from human-reviewed results

## Tech Stack

- Python
- Ollama (Llama 3.2)
- Pandas
- Jupyter Notebook

## Project Structure

```text
Sentiment-LLM-Evaluation/
│
├── main.py
├── config.py
├── prompts.py
├── llm.py
├── evaluation.py
├── review.py
├── requirements.txt
├── README.md
│
├── data/
│   └── db2.csv
│
├── outputs/
│   ├── final_results_20260809_115549.csv
│   ├── gold_dataset.csv
│   ├── human_review_queue.csv
│   └── human_reviewed.csv
│
└── screenshots/
    ├── img1.png
    └── img2.png
