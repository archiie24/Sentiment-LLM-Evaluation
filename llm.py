"""
Functions for interacting with the local LLM (Ollama).
"""

import json

from ollama import chat

from config import MODEL
from prompts import (
    SYSTEM_MSG,
    PROMPT_A,
    PROMPT_B,
    JUDGE_PROMPT,
)


# ==========================================================
# Build Batch Prompt
# ==========================================================

def build_batched_prompt(email_dict):
    """
    Creates one prompt containing multiple emails.

    Each email is evaluated using:
    1. Prompt A
    2. Prompt B
    3. Judge comparison

    All three operations happen in ONE LLM call.
    """

    prompt = f"""
{PROMPT_A}

{PROMPT_B}

{JUDGE_PROMPT}

For EACH email, return exactly this JSON structure:

{{
    "<email_id>": {{
        "A": {{
            "sentiment": "positive|negative|neutral",
            "confidence": 0.0,
            "reason": "short supporting quote"
        }},
        "B": {{
            "sentiment": "positive|negative|neutral",
            "confidence": 0.0,
            "reason": "short supporting quote"
        }},
        "judge": {{
            "winner": "A|B|same",
            "suggested_sentiment": "positive|negative|neutral",
            "explanation": "short explanation"
        }}
    }}
}}

Return ONLY valid JSON.
Do not include markdown.
Do not include text outside the JSON.

Emails:
"""

    for eid, text in email_dict.items():

        clean = str(text).replace("\n", " ")

        prompt += (
            f"\nEmail ID: {eid}\n"
            f"Email: {clean}\n"
        )

    return prompt


# ==========================================================
# Parse JSON Safely
# ==========================================================

def parse_json(content):
    """
    Attempts to parse JSON even if the model
    adds extra text.
    """

    try:

        return json.loads(content)

    except Exception:

        try:

            start = content.index("{")
            end = content.rindex("}") + 1

            return json.loads(
                content[start:end]
            )

        except Exception:

            print(
                "\nInvalid JSON returned by model:\n"
            )

            print(content)

            return {}


# ==========================================================
# Single LLM Call
# ==========================================================

def call_llm_json(prompt):
    """
    Sends one prompt to Ollama and returns parsed JSON.
    """

    response = chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_MSG,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    content = response["message"]["content"].strip()

    return parse_json(content)


# ==========================================================
# Batch Processing
# ==========================================================

def run_batched_llm(df, batch_size=20):
    """
    Performs batch inference.

    For every batch:
        - Prompt A
        - Prompt B
        - Judge

    are performed inside ONE LLM request.
    """

    all_results = {}

    total = len(df)

    for start in range(0, total, batch_size):

        end = min(
            start + batch_size,
            total
        )

        print(
            f"Processing rows {start} - {end - 1}"
        )

        chunk = df.iloc[start:end]

        emails = {}

        for idx, row in chunk.iterrows():

            emails[str(idx)] = str(
                row["email_text"]
            )

        # One LLM call for the entire batch
        prompt = build_batched_prompt(emails)

        batch_results = call_llm_json(prompt)

        all_results.update(batch_results)

    return all_results
