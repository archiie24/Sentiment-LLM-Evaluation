# LLM Evaluation Framework

A Treater-inspired framework for evaluating customer email sentiment using a local LLM (Ollama + Llama 3.2). The project compares multiple prompting strategies, validates outputs using rule-based guardrails, and supports human-in-the-loop review for reliable sentiment classification.

---

## Features

- Dual-prompt sentiment evaluation
- Batch inference with Ollama (Llama 3.2)
- Confidence-based output selection
- Rule-based guardrail validation
- Human review queue generation
- Final approved sentiment export

---

## Tech Stack

- Python
- Ollama (Llama 3.2)
- Pandas
- Jupyter Notebook

---

## Project Structure

```
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
```

---

## Pipeline

```
Customer Emails
      │
      ▼
Prompt A & Prompt B
      │
      ▼
Confidence Comparison
      │
      ▼
Guardrail Validation
      │
      ▼
Human Review (if needed)
      │
      ▼
Final Approved Output
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/LLM-Evaluation-Framework.git
cd LLM-Evaluation-Framework
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the model:

```bash
ollama pull llama3.2
```

Run the project:

```bash
python main.py
```

---

## Future Improvements

- Multi-model evaluation
- LLM-as-a-Judge scoring
- Dashboard for evaluation metrics
- RAG integration
