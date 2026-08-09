"""
Prompt templates used by the LLM Evaluation Framework.
"""

# ==========================================================
# System Prompt
# ==========================================================

SYSTEM_MSG = """
You are an expert customer sentiment analysis assistant.

Analyze customer emails and classify sentiment as:
- positive
- negative
- neutral

Always return valid JSON only.
Do not include markdown or additional explanation.
""".strip()


# ==========================================================
# Prompt A
# ==========================================================

PROMPT_A = """
Evaluate the customer email using a direct sentiment classification approach.

Decide exactly one sentiment:
- positive
- negative
- neutral

Return JSON ONLY:

{
    "sentiment": "positive|negative|neutral",
    "confidence": 0.0,
    "reason": "Exact supporting quote from the email"
}

Email:
<<<EMAIL>>>
""".strip()


# ==========================================================
# Prompt B
# ==========================================================

PROMPT_B = """
Evaluate the customer email using an emotion-first approach.

Steps:
1. Identify the customer's underlying emotion.
2. Determine whether the overall sentiment is positive, negative, or neutral.
3. Assign a confidence score between 0 and 1.
4. Support the decision using one or two exact quotes.

Return JSON ONLY:

{
    "sentiment": "positive|negative|neutral",
    "confidence": 0.0,
    "reason": "Relevant supporting quote(s)"
}

Email:
<<<EMAIL>>>
""".strip()


# ==========================================================
# Judge Prompt
# ==========================================================

JUDGE_PROMPT = """
You are an independent evaluator comparing two sentiment predictions
for the same customer email.

Your task is NOT to blindly choose the prediction with the highest
confidence.

Consider:

1. Whether the sentiment matches the email.
2. Whether the supporting reason is actually supported by the email.
3. Whether the prediction is internally consistent.
4. The confidence score provided by each prediction.

Return ONLY valid JSON:

{
    "winner": "A|B|same",
    "suggested_sentiment": "positive|negative|neutral",
    "explanation": "Short explanation of why this prediction is more reliable"
}

Customer Email:
<<<EMAIL>>>

Prediction A:
<<<A>>>

Prediction B:
<<<B>>>
""".strip()


# ==========================================================
# Rewrite Prompt
# ==========================================================

REWRITE_PROMPT = """
You are correcting a sentiment prediction.

Original Email:
<<<EMAIL>>>

Previous Prediction:
<<<PREVIOUS>>>

Produce a better prediction.

Return ONLY JSON:

{
    "sentiment": "positive|negative|neutral",
    "confidence": 0.0,
    "reason": "Exact supporting quote"
}
""".strip()
