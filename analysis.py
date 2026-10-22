import csv
import json
from pathlib import Path
from collections import Counter

DATA_DIR = Path(__file__).parent / "data"

CSV_FILES = ["shops.csv", "products.csv", "offers.csv"]


def sample_csv(path, n=5):
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = []
        for i, row in enumerate(reader):
            rows.append(row)
            if i >= n:
                break
    return rows


def analyze_csv(path):
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        total = 0
        null_counts = Counter()
        sample = []
        for i, r in enumerate(reader):
            total += 1
            for h in headers:
                v = r.get(h)
                if v is None or v == "" or v.strip() == "{}" or v.strip() == "[]" or v.strip().lower() == "null":
                    null_counts[h] += 1
            if i < 3:
                sample.append(r)
    return {
        "path": str(path),
        "headers": headers,
        "total_rows": total,
        "null_counts": dict(null_counts),
        "sample_rows": sample
    }


if __name__ == "__main__":
    report = {}
    for fname in CSV_FILES:
        p = DATA_DIR / fname
        if not p.exists():
            # avoid printing to console; use logging if needed
            # print(f"Missing {p}")
            continue
        # print(f"Analyzing {p} ...")
        report[fname] = analyze_csv(p)
    # results are returned; caller may log or print as needed
    # print(json.dumps(report, ensure_ascii=False, indent=2))
