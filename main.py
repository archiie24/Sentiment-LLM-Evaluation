"""
Main entry point for the LLM Evaluation Framework.
"""

import pandas as pd

from config import (
    CSV_PATH,
    BATCH_SIZE,
)

from llm import run_batched_llm

from evaluation import (
    results_to_dataframe,
    add_review_flags,
    summarize_results,
    confidence_statistics,
)

from review import (
    export_human_review_queue,
    apply_human_review,
    export_final_results,
    extract_new_gold_labels,
    save_gold_dataset,
)


def load_dataset():
    """
    Load and preprocess the input dataset.
    """

    print("Loading dataset...")

    df = pd.read_csv(CSV_PATH)

    df.columns = [c.strip() for c in df.columns]

    df["email_text"] = (
        df["email_text"]
        .astype(str)
        .str.strip()
    )

    if "gold_reason" in df.columns:

        df["gold_reason"] = (

            df["gold_reason"]

            .astype(str)

            .str.lower()

            .str.strip()

            .replace({

                "": None,

                "nan": None

            })

        )

    print(f"Loaded {len(df)} emails.")

    return df


def main():

    # -----------------------------------------------------
    # Load Dataset
    # -----------------------------------------------------

    df = load_dataset()

    # -----------------------------------------------------
    # Run LLM
    # -----------------------------------------------------

    print()

    print("=" * 60)

    print("Running LLM Evaluation...")

    print("=" * 60)

    batched_results = run_batched_llm(

        df,

        batch_size=BATCH_SIZE

    )

    # -----------------------------------------------------
    # Convert Results
    # -----------------------------------------------------

    results_df = results_to_dataframe(

        batched_results,

        df

    )

    # -----------------------------------------------------
    # Review Flags
    # -----------------------------------------------------

    results_df = add_review_flags(results_df)

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    summarize_results(results_df)

    stats = confidence_statistics(results_df)

    print()

    print("Confidence Statistics")

    print("---------------------")

    for k, v in stats.items():

        print(f"{k}: {v}")

    # -----------------------------------------------------
    # Export Review Queue
    # -----------------------------------------------------

    export_human_review_queue(results_df)

    # -----------------------------------------------------
    # Apply Human Review
    # -----------------------------------------------------

    results_df = apply_human_review(results_df)

    # -----------------------------------------------------
    # Export Final Results
    # -----------------------------------------------------

    export_final_results(results_df)

    # -----------------------------------------------------
    # Gold Dataset
    # -----------------------------------------------------

    gold = extract_new_gold_labels(results_df)

    save_gold_dataset(gold)

    print()

    print("=" * 60)

    print("Pipeline Complete")

    print("=" * 60)


if __name__ == "__main__":

    main()