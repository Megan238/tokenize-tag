import csv
import json
import argparse
from tag import tokenize_and_tag_batch


def main(input_csv: str, output_jsonl: str):
    results = []

    with open(input_csv, newline="", encoding="gb18030") as f:
        reader = csv.DictReader(f)
        assert "search_term" in reader.fieldnames
        assert "language" in reader.fieldnames
        
        keywords = []
        for i, row in enumerate(reader):
            if i == 50 :
                break
            keyword = (row["search_term"] or "").strip()

            if not keyword:
                continue
            keywords.append(keyword)
        
        results = tokenize_and_tag_batch(keywords)

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


    print(f"Processed {len(results)} rows")
    print(f"Output written to {output_jsonl}")


if __name__ == "__main__":
    main("keywords.csv", "keywords_test_output.json")
