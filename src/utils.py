import pandas as pd
import os

SEGMENT_LABELS = {
    0: "Budget Shoppers",
    1: "High-Value Loyalists",
    2: "Young Explorers",
    3: "Occasional Buyers",
    4: "Premium Spenders",
}


def label_segments(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Segment_Label"] = df["Cluster"].map(lambda x: SEGMENT_LABELS.get(x, f"Segment {x}"))
    return df


def save_segmented_output(df: pd.DataFrame, path: str = "outputs/segmented_customers.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved segmented data → {path}")


def print_summary(summary: pd.DataFrame):
    print("\n" + "=" * 55)
    print("  CUSTOMER SEGMENT SUMMARY")
    print("=" * 55)
    print(summary.to_string())
    print("=" * 55 + "\n")
