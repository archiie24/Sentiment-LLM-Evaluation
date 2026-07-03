"""
Functions for interacting with the local LLM (Ollama).
"""

import json
from ollama import chat

from config import MODEL
from prompts import SYSTEM_MSG


# ==========================================================
# Build Batch Prompt
# ==========================================================

def build_batched_prompt(email_dict):
    """
    Creates a prompt containing multiple emails.

    Parameters
    ----------
    email_dict : dict
        Mapping of email_id -> email_text

    Returns
    -------
    str
        Prompt sent to the LLM.
    """

    prompt = """
Evaluate every email below.

For EACH email return JSON in exactly this format:

{
    "<email_id>":{
        "A":{
            "sentiment":"",
            "confidence":0.0,
            "reason":""
        },
        "B":{
            "sentiment":"",
            "confidence":0.0,
            "reason":""
        },
        "judge":{
            "suggested_sentiment":"",
            "difference":"",
            "explanation":""
        }
    }
}

Return ONLY valid JSON.
No markdown.
No explanation outside JSON.

Emails:

"""

    for eid, text in email_dict.items():

        clean = str(text).replace("\n", " ")

        prompt += f"\nEmail ID: {eid}\n"

        prompt += f"Email: {clean}\n"

    return prompt


# ==========================================================
# Parse JSON Safely
# ==========================================================

def parse_json(content):
    """
    Attempts to parse JSON even if the model adds extra text.
    """

    try:
        return json.loads(content)

    except Exception:

        try:

            start = content.index("{")

            end = content.rindex("}") + 1

            return json.loads(content[start:end])

        except Exception:

            print("\nInvalid JSON returned by model:\n")

            print(content)

            return {}


# ==========================================================
# Single LLM Call
# ==========================================================

def call_llm_json(prompt):
    """
    Sends prompt to Ollama and returns parsed JSON.
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
    Performs batch inference over the dataframe.

    Parameters
    ----------
    df : pandas.DataFrame

    batch_size : int

    Returns
    -------
    dict
    """

    all_results = {}

    total = len(df)

    for start in range(0, total, batch_size):

        end = min(start + batch_size, total)

        print(f"Processing rows {start} - {end-1}")

        chunk = df.iloc[start:end]

        emails = {}

        for idx, row in chunk.iterrows():

            emails[str(idx)] = str(row["email_text"])

        prompt = build_batched_prompt(emails)

        batch_results = call_llm_json(prompt)

        all_results.update(batch_results)

    return all_results