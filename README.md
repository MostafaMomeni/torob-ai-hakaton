This package contains the Data Access / Repository layer for the shopping-assistant project.

Files:
- analysis.py        # dataset inspection report (reads CSVs and prints summary)
- schemas.py         # Pydantic models mirroring CSV columns + internal fields
- repository.py      # repository API: load CSV, query functions, integrity checks
- validators.py      # data validation helpers
- freshness.py       # freshness / staleness helpers
- tests/test_data_quality.py  # basic tests for missing/invalid/duplicate data

Source CSVs: expected under `data/` with filenames exactly:
- shops.csv
- products.csv
- offers.csv

Mapping of CSV columns to internal fields is documented in README_fields.md

Run `python analysis.py` to generate a dataset report.
