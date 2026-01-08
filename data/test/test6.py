import json

INPUT_FILE = "data/jsonFiler/judged_clean_rag.json"
OUTPUT_FILE = "data/jsonFiler/errors_with_correct_chunks.json"

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    bad_cases = [
        {
            "question_id": x["question_id"],
            "correctness": x["correctness"],
            "correct_chunks": x["correct_chunks"],
            "justification": x.get("justification", {})
        }
        for x in data
        if int(x.get("correct_chunks", 0)) == 1 and int(x.get("correctness", 0)) == 0
    ]

    print(f"Antal frågor med rätt chunks men fel svar: {len(bad_cases)}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(bad_cases, f, indent=2, ensure_ascii=False)

    print(f"Sparade resultat till {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
# Run code with: docker compose exec app python data/test/test6.py
