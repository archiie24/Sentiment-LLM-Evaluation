"""
Configuration settings for the LLM Evaluation Framework.
"""

# -------------------------------
# Model Configuration
# -------------------------------

MODEL = "llama3.2"

# -------------------------------
# Dataset
# -------------------------------

CSV_PATH = "data/sample_db2.csv"

# -------------------------------
# Batch Processing
# -------------------------------

BATCH_SIZE = 20

# -------------------------------
# Review Thresholds
# -------------------------------

LOW_CONFIDENCE_THRESHOLD = 0.60

# -------------------------------
# Output Paths
# -------------------------------

HUMAN_REVIEW_FILE = "outputs/human_review_queue.csv"

FINAL_OUTPUT_DIR = "outputs"

HUMAN_REVIEWED_FILE = "outputs/human_reviewed.csv"