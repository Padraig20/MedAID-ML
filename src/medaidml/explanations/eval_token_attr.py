import pandas as pd
import numpy as np
import argparse
import os

from medaidml import RESULTS_DIR

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run analysis on IG-extracted token attributions")
    parser.add_argument("--top_k", type=int, default=10, help="Number of top tokens to display")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    TOP_K = args.top_k

    token_attributions_df = pd.read_csv(os.path.join(RESULTS_DIR, "token_attributions.csv"))

    for language in ['en', 'de', 'es', 'fr']:
        print(f"Top {TOP_K} tokens for language: {language}")
        language_token_scores = {
            row['token']: np.mean(row['score'])
            for _, row in token_attributions_df.iterrows()
            if row['language'] == language
        }
        sorted_tokens = sorted(language_token_scores.items(), key=lambda x: x[1], reverse=True)[:TOP_K]
        for token, score in sorted_tokens:
            print(f"{token:15s} -> {score:.4f}")
        print("-" * 30)