#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd


def extract_problem(messages):
    if not isinstance(messages, list) or not messages:
        return ""

    convo = messages[0] if isinstance(messages[0], list) else messages
    if not isinstance(convo, list):
        return ""

    for message in convo:
        if isinstance(message, dict) and message.get("role") == "user":
            return str(message.get("content", ""))
    return ""


def convert_record(record: dict) -> dict:
    return {
        "problem": extract_problem(record.get("messages")),
        "answer": record.get("solution", ""),
        "images": record.get("images", []),
        "metadata": {
            "rm_type": "markdown_table",
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Convert markdown table RL JSON data to slime parquet format.")
    parser.add_argument("-i", "--input", required=True, help="Input JSON path.")
    parser.add_argument("-o", "--output", required=True, help="Output parquet path.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected top-level list, got {type(data).__name__}")

    converted = [convert_record(record) for record in data]
    df = pd.DataFrame(converted)
    df.to_parquet(output_path, index=False)

    print(f"Converted {len(df)} samples to {output_path}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
