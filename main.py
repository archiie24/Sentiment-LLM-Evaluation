"""
Main entry point for the LLM Evaluation Framework.
"""

import os
import json
import pandas as pd

from config import CSV_PATH, BATCH_SIZE

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


PENDING_RESULTS_PATH = "outputs/pending_llm_results.json"


def load_dataset():
    print("Loading dataset...")

    df = pd.read_csv(CSV_PATH)

    df.columns = [c.strip() for c in df.columns]

    if "id" in df.columns:
        df["id"] = pd.to_numeric(
            df["id"],
            errors="coerce"
        )

        df = df.dropna(subset=["id"])

        df["id"] = df["id"].astype(int)

        df = df.set_index(
            "id",
            drop=False
        )

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


def get_llm_results(df):

    if os.path.exists(PENDING_RESULTS_PATH):

        print()
        print("Existing LLM results found.")
        print("Loading saved results...")

        with open(
            PENDING_RESULTS_PATH,
            "r",
            encoding="utf-8"
        ) as f:
            return json.load(f)

    print()
    print("=" * 60)
    print("Running LLM Evaluation...")
    print("=" * 60)

    results = run_batched_llm(
        df,
        batch_size=BATCH_SIZE
    )

    os.makedirs(
        "outputs",
        exist_ok=True
    )

    with open(
        PENDING_RESULTS_PATH,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            results,
            f,
            indent=2
        )

    print()
    print(
        f"LLM results saved to: "
        f"{PENDING_RESULTS_PATH}"
    )

    return results


def main():

    df = load_dataset()

    batched_results = get_llm_results(df)

    results_df = results_to_dataframe(
        batched_results,
        df
    )

    results_df = add_review_flags(
        results_df
    )

    summarize_results(
        results_df
    )

    stats = confidence_statistics(
        results_df
    )

    print()
    print("Confidence Statistics")
    print("---------------------")

    for key, value in stats.items():
        print(f"{key}: {value}")

    export_human_review_queue(
        results_df
    )

    results_df = apply_human_review(
        results_df
    )

    export_final_results(
        results_df
    )

    gold = extract_new_gold_labels(
        results_df
    )

    save_gold_dataset(
        gold
    )

    print()
    print("=" * 60)
    print("Pipeline Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
