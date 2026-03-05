"""Download ScienceQA from HuggingFace and convert to VisAttnSink format.

Outputs:
  {output_dir}/
    Images/          - saved images (only for samples that have images)
    Questions/
      test-questions.jsonl   - for src/inference.py
    problems.json    - for eval_science_qa.py
    pid_splits.json  - for eval_science_qa.py
"""
import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def format_question_with_choices(question, choices, hint=""):
    """Format a multiple-choice question with lettered options."""
    options = "ABCDE"
    parts = []
    if hint:
        parts.append(f"Context: {hint}")
    parts.append(question)
    for i, choice in enumerate(choices):
        parts.append(f"{options[i]}. {choice}")
    parts.append("Answer with the option's letter from the given choices directly.")
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    img_dir = output_dir / "Images"
    q_dir = output_dir / "Questions"
    img_dir.mkdir(parents=True, exist_ok=True)
    q_dir.mkdir(parents=True, exist_ok=True)

    print("Loading ScienceQA dataset...")
    ds = load_dataset("derek-thomas/ScienceQA")

    # We need problems.json and pid_splits.json for eval_science_qa.py
    problems = {}
    pid_splits = {"train": [], "val": [], "test": []}

    # Build problems.json from all splits
    global_id = 0
    id_map = {}  # (split, local_idx) -> global_id string

    for split_name in ["train", "validation", "test"]:
        eval_split = "val" if split_name == "validation" else split_name
        for idx, row in enumerate(tqdm(ds[split_name], desc=f"Indexing {split_name}")):
            sid = str(global_id)
            id_map[(split_name, idx)] = sid
            pid_splits[eval_split].append(sid)
            problems[sid] = {
                "question": row["question"],
                "choices": row["choices"],
                "answer": row["answer"],
                "hint": row["hint"],
                "subject": row["subject"],
                "grade": row["grade"],
                "has_image": row["image"] is not None,
            }
            global_id += 1

    # Save problems.json and pid_splits.json
    with open(output_dir / "problems.json", "w") as f:
        json.dump(problems, f)
    with open(output_dir / "pid_splits.json", "w") as f:
        json.dump(pid_splits, f)
    print(f"Saved problems.json ({len(problems)} problems) and pid_splits.json")

    # Now process the target split: save images and create JSONL
    split_data = ds[args.split]
    split_key = "validation" if args.split == "val" else args.split
    questions = []
    skipped = 0

    for idx, row in enumerate(tqdm(split_data, desc=f"Processing {args.split}")):
        sid = id_map[(split_key, idx)]
        image = row["image"]

        # Skip samples without images (text-only questions)
        if image is None:
            skipped += 1
            continue

        # Save image
        img_filename = f"{sid}.png"
        img_path = img_dir / img_filename
        if not img_path.exists():
            image.save(img_path)

        # Format question
        text = format_question_with_choices(
            row["question"], row["choices"], row.get("hint", "")
        )

        options = "ABCDE"
        questions.append({
            "qid": sid,
            "question_id": sid,
            "image": img_filename,
            "text": text,
            "question": text,
            "label": options[row["answer"]],
            "gt-label": options[row["answer"]],
        })

    # Save JSONL
    jsonl_path = q_dir / f"{args.split}-questions.jsonl"
    with open(jsonl_path, "w") as f:
        for q in questions:
            f.write(json.dumps(q) + "\n")

    print(f"\nDone! {len(questions)} image questions saved, {skipped} text-only skipped")
    print(f"  Images: {img_dir}/")
    print(f"  Questions: {jsonl_path}")
    print(f"  problems.json: {output_dir / 'problems.json'}")
    print(f"  pid_splits.json: {output_dir / 'pid_splits.json'}")


if __name__ == "__main__":
    main()
