"""
Prompt templates used by the LLM Evaluation Framework.
"""

# ==========================================================
# System Prompt
# ==========================================================

SYSTEM_MSG = """
You are an expert customer sentiment analysis assistant.

Your task is to analyze customer emails and return structured JSON only.

For every email:
1. Evaluate using Prompt A.
2. Evaluate using Prompt B.
3. Compare both predictions.
4. Select the better prediction based on confidence.

Return ONLY valid JSON.
Do not include markdown.
Do not include explanations outside JSON.
""".strip()


# ==========================================================
# Prompt A
# ==========================================================

PROMPT_A = """
You are a customer-experience specialist.

Read the email below.

Decide EXACTLY one sentiment label.

Possible labels:

- positive
- negative
- neutral

Return JSON ONLY.

{
    "sentiment":"positive|negative|neutral",
    "confidence":0.0,
    "reason":"Exact quote from the email"
}

Email:

<<<EMAIL>>>
""".strip()


# ==========================================================
# Prompt B
# ==========================================================

PROMPT_B = """
You are an expert sentiment evaluator.

Instructions:

1. Identify the customer's emotion.
2. Decide the sentiment.
3. Assign a confidence score between 0 and 1.
4. Support the decision using one or two exact quotes from the email.

Return ONLY JSON.

{
    "sentiment":"positive|negative|neutral",
    "confidence":0.0,
    "reason":"Relevant quote(s)"
}

Email:

<<<EMAIL>>>
""".strip()


# ==========================================================
# Judge Prompt
# ==========================================================

JUDGE_PROMPT = """
You are comparing two sentiment predictions.

Prediction A:
<<<A>>>

Prediction B:
<<<B>>>

Choose whichever prediction is more reliable.

Consider:

- confidence
- quality of supporting reason
- consistency

Return ONLY JSON.

{
    "winner":"A|B|same",
    "explanation":"short explanation"
}
""".strip()


# ==========================================================
# Rewrite Prompt
# ==========================================================

REWRITE_PROMPT = """
You are correcting a sentiment prediction.

Original Email

<<<EMAIL>>>

Previous Prediction

<<<PREVIOUS>>>

Produce a better prediction.

Return ONLY JSON.

{
    "sentiment":"positive|negative|neutral",
    "confidence":0.0,
    "reason":"Exact supporting quote"
}
""".strip()