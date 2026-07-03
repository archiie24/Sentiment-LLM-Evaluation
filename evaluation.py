"""
Evaluation utilities for the LLM Evaluation Framework.
"""

import pandas as pd
from config import LOW_CONFIDENCE_THRESHOLD


# ==========================================================
# Convert LLM Results to DataFrame
# ==========================================================

def results_to_dataframe(results_dict, df):
    """
    Converts the nested LLM output dictionary into a DataFrame.
    """

    rows = []

    for eid, content in results_dict.items():

        A = content.get("A", {})
        B = content.get("B", {})
        J = content.get("judge", {})

        email_text = df.loc[int(eid), "email_text"]

        gold = None
        if "gold_reason" in df.columns:
            gold = df.loc[int(eid), "gold_reason"]

        A_conf = float(A.get("confidence", 0) or 0)
        B_conf = float(B.get("confidence", 0) or 0)

        # -------------------------------
        # Select higher confidence result
        # -------------------------------

        if A_conf > B_conf:

            final_label = A.get("sentiment")
            selected = "A"

        elif B_conf > A_conf:

            final_label = B.get("sentiment")
            selected = "B"

        else:

            final_label = A.get("sentiment")
            selected = "same_confidence"

        rows.append({

            "id": eid,

            "email": email_text,

            "gold_reason": gold,

            "A_sentiment": A.get("sentiment"),
            "A_confidence": A_conf,
            "A_reason": A.get("reason"),

            "B_sentiment": B.get("sentiment"),
            "B_confidence": B_conf,
            "B_reason": B.get("reason"),

            "judge_suggested": J.get("suggested_sentiment"),
            "judge_difference": J.get("difference"),
            "judge_explanation": J.get("explanation"),

            "final_sentiment": final_label,
            "selected_by": selected

        })

    return pd.DataFrame(rows)


# ==========================================================
# Review Flags
# ==========================================================

def add_review_flags(results_df,
                     low_conf=LOW_CONFIDENCE_THRESHOLD):
    """
    Flags uncertain predictions for manual review.
    """

    def needs_review(row):

        if row["A_confidence"] < low_conf and \
           row["B_confidence"] < low_conf:

            return "Both confidences low"

        if row["A_sentiment"] != row["B_sentiment"]:

            return "Prompt A and Prompt B disagree"

        return None

    results_df["review_reason"] = results_df.apply(
        needs_review,
        axis=1
    )

    results_df["needs_review"] = (
        results_df["review_reason"].notnull()
    )

    return results_df


# ==========================================================
# Dataset Summary
# ==========================================================

def summarize_results(results_df):
    """
    Prints summary statistics.
    """

    print("=" * 50)

    print("Evaluation Summary")

    print("=" * 50)

    print()

    print("Total Emails:")

    print(len(results_df))

    print()

    print("Needs Review:")

    print(results_df["needs_review"].sum())

    print()

    print("Selected By")

    print(results_df["selected_by"].value_counts())

    print()

    print("Final Sentiment Distribution")

    print(results_df["final_sentiment"].value_counts())

    print("=" * 50)


# ==========================================================
# Confidence Statistics
# ==========================================================

def confidence_statistics(results_df):
    """
    Returns average confidence values.
    """

    return {

        "Average A Confidence":
            results_df["A_confidence"].mean(),

        "Average B Confidence":
            results_df["B_confidence"].mean(),

        "Higher A":
            (results_df["selected_by"] == "A").sum(),

        "Higher B":
            (results_df["selected_by"] == "B").sum()

    }