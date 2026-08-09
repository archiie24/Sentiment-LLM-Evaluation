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
LLM-Evaluation-Framework/
│
├── main.py
├── config.py
├── prompts.py
├── llm.py
├── evaluation.py
├── review.py
├── data/
├── outputs/
├── notebooks/
├── README.md
└── requirements.txt
