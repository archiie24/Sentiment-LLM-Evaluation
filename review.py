"""
Human review utilities for the LLM Evaluation Framework.
"""

import os
from datetime import datetime

import pandas as pd

from config import (
    HUMAN_REVIEW_FILE,
    HUMAN_REVIEWED_FILE,
    FINAL_OUTPUT_DIR,
)


# ==========================================================
# Export Human Review Queue
# ==========================================================

def export_human_review_queue(
    results_df,
    out_path=HUMAN_REVIEW_FILE
):
    """
    Export uncertain predictions for manual review.
    """

    review_df = results_df[
        results_df["needs_review"]
    ].copy()

    columns = [

        "id",

        "email",

        "final_sentiment",

        "A_sentiment",
        "A_confidence",

        "B_sentiment",
        "B_confidence",

        "review_reason"

    ]

    columns = [
        c for c in columns
        if c in review_df.columns
    ]

    review_df[columns].to_csv(
        out_path,
        index=False
    )

    print(f"Human review queue saved to: {out_path}")

    return out_path


# ==========================================================
# Apply Human Review
# ==========================================================

def apply_human_review(
    results_df,
    reviewed_csv=HUMAN_REVIEWED_FILE
):
    """
    Merge human-reviewed labels with model predictions.
    """

    if not os.path.exists(reviewed_csv):

        print("No reviewed file found.")

        return results_df

    reviewed = pd.read_csv(reviewed_csv)

    reviewed = reviewed[
        [
            "id",
            "human_final_sentiment",
            "human_notes"
        ]
    ]

    merged = results_df.merge(

        reviewed,

        on="id",

        how="left"

    )

    merged["final_sentiment"] = (

        merged["human_final_sentiment"]

        .fillna(merged["final_sentiment"])

    )

    merged["review_source"] = (

        merged["human_final_sentiment"]

        .apply(

            lambda x:
            "human"

            if pd.notnull(x)

            else "model"

        )

    )

    print("Human review applied.")

    return merged


# ==========================================================
# Export Final Results
# ==========================================================

def export_final_results(
    results_df,
    output_dir=FINAL_OUTPUT_DIR
):
    """
    Export the final approved dataset.
    """

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    filename = (

        f"final_results_{timestamp}.csv"

    )

    path = os.path.join(
        output_dir,
        filename
    )

    columns = [

        "email",

        "final_sentiment",

        "review_source",

        "A_confidence",

        "B_confidence"

    ]

    columns = [

        c for c in columns

        if c in results_df.columns

    ]

    results_df[columns].to_csv(

        path,

        index=False

    )

    print(f"Final results exported to: {path}")

    return path


# ==========================================================
# Extract New Gold Labels
# ==========================================================

def extract_new_gold_labels(results_df):
    """
    Extract human-approved labels to build
    future gold datasets.
    """

    if "review_source" not in results_df.columns:

        print("No human reviews available.")

        return None

    gold = results_df[

        results_df["review_source"] == "human"

    ][

        [

            "email",

            "final_sentiment"

        ]

    ]

    return gold


# ==========================================================
# Save Gold Dataset
# ==========================================================

def save_gold_dataset(
    gold_df,
    output_dir=FINAL_OUTPUT_DIR
):
    """
    Save extracted gold labels.
    """

    if gold_df is None:

        return None

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    path = os.path.join(

        output_dir,

        "gold_dataset.csv"

    )

    gold_df.to_csv(

        path,

        index=False

    )

    print(f"Gold dataset saved to: {path}")

    return path