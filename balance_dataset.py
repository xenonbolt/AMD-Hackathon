"""
Dataset Re-Balancer for Vulnerability Training Data.

Problem: 50/50 positive/negative ratio causes the model to default to
predicting "no vulnerabilities" (the easier class). This script reduces
the negative ratio to a target percentage (default 20%).

Usage:
    python balance_dataset.py \
        --input "Dataset/train_classifier_precise_lines.jsonl" \
        --output "Dataset/train_balanced.jsonl" \
        --neg_ratio 0.20
"""
import argparse
import json
import logging
import random
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("balance_dataset")


def balance_dataset(input_path: str, output_path: str, neg_ratio: float = 0.20, seed: int = 42):
    """
    Reads the training JSONL, separates positive (has vulns) and negative (no vulns)
    examples, downsamples negatives to the target ratio, shuffles, and writes the result.
    """
    positives = []
    negatives = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            text = data.get("text", "")
            # Check if this is a negative example (empty vulnerability list)
            if '"vulnerabilities": []' in text or '"vulnerabilities":[]' in text:
                negatives.append(line.strip())
            else:
                positives.append(line.strip())

    logger.info(f"Original dataset: {len(positives)} positive, {len(negatives)} negative ({len(positives)+len(negatives)} total)")

    # Calculate how many negatives to keep
    # target_neg / (num_pos + target_neg) = neg_ratio
    # target_neg = neg_ratio * num_pos / (1 - neg_ratio)
    num_pos = len(positives)
    target_neg_count = int(neg_ratio * num_pos / (1.0 - neg_ratio))
    target_neg_count = min(target_neg_count, len(negatives))  # can't exceed available

    logger.info(f"Target negative ratio: {neg_ratio*100:.0f}%")
    logger.info(f"Keeping {target_neg_count} of {len(negatives)} negative examples")

    # Downsample negatives
    random.seed(seed)
    sampled_negatives = random.sample(negatives, target_neg_count)

    # Combine and shuffle
    balanced = positives + sampled_negatives
    random.shuffle(balanced)

    # Write output
    output_file = Path(output_path)
    with open(output_file, "w", encoding="utf-8") as f:
        for line in balanced:
            f.write(line + "\n")

    total = len(balanced)
    actual_neg_ratio = target_neg_count / total * 100
    logger.info(f"Balanced dataset: {num_pos} positive, {target_neg_count} negative ({total} total)")
    logger.info(f"Actual negative ratio: {actual_neg_ratio:.1f}%")
    logger.info(f"Saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-balance training dataset by downsampling negatives")
    parser.add_argument("--input", type=str, default="Dataset/train_classifier_precise_lines.jsonl",
                        help="Input JSONL training file")
    parser.add_argument("--output", type=str, default="Dataset/train_balanced.jsonl",
                        help="Output balanced JSONL file")
    parser.add_argument("--neg_ratio", type=float, default=0.20,
                        help="Target negative ratio (default: 0.20 = 20%%)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    balance_dataset(args.input, args.output, args.neg_ratio, args.seed)
