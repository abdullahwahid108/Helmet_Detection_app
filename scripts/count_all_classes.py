import argparse
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    ap.add_argument("--data-root", type=str, default="data")
    args = ap.parse_args()

    lbl_dir = Path(args.data_root) / "labels" / args.split
    counts = Counter()
    files_with_issues = []

    for txt_path in lbl_dir.glob("*.txt"):
        for line in txt_path.read_text().strip().splitlines():
            if not line.strip():
                continue
            tokens = line.split()
            if len(tokens) != 5:
                files_with_issues.append((txt_path.name, len(tokens)))
                continue
            cls = int(float(tokens[0]))
            counts[cls] += 1

    print(f"Class index counts in {args.split}:")
    for cls in sorted(counts.keys()):
        print(f"  class {cls}: {counts[cls]} boxes")

    if files_with_issues:
        print(f"\n{len(files_with_issues)} lines with unexpected token count (not 5):")
        for name, n in files_with_issues[:10]:
            print(f"  {name}: {n} tokens")


if __name__ == "__main__":
    main()