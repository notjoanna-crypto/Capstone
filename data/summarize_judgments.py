import json
from collections import Counter

# docker compose exec app python data/summarize_judgments.py


INPUT_FILE = "data/jsonFiler/judged_conflict_k20.json"

METRICS = ["correctness", "hallucination", "source_drift", "correct_chunks"]

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON-filen måste innehålla en lista med objekt.")

    n = len(data)
    if n == 0:
        print("Filen innehåller 0 poster.")
        return

    # Summeringar
    sums = {
        m: sum(int(x.get(m, 0)) for x in data)
        for m in METRICS
    }

    # Fördelningar
    dist = {
        m: Counter(int(x.get(m, 0)) for x in data)
        for m in METRICS
    }

    # Rates
    rates = {m: sums[m] / n for m in METRICS}

    print(f"Antal frågor: {n}\n")

    for m in METRICS:
        ones = sums[m]
        zeros = dist[m].get(0, 0)
        print(f"{m}:")
        print(f"  1: {ones}  ({ones/n:.3%})")
        print(f"  0: {zeros} ({zeros/n:.3%})")
        print()

    # Kombinationer
    combos = Counter(
        tuple(int(x.get(m, 0)) for m in METRICS)
        for x in data
    )

if __name__ == "__main__":
    main()
