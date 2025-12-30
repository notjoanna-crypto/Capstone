import os
import json
import time
from datetime import datetime, timezone
from dotenv import load_dotenv
from datapizza.clients.openai import OpenAIClient

# Run code with: docker compose exec app python data/WithoutRAG.py

INPUT_PATH = "data/ground_truth_final.json"   # <-- ändra till din fil
OUTPUT_PATH = "data/without_rag_answers.json"     # <-- outputfil

MODEL = "gpt-4o-mini"
SYSTEM_PROMPT = "You are a helpful assistant that can answer questions and help with tasks."

SLEEP_SECONDS_BETWEEN_CALLS = 0.0  # sätt t.ex. 0.2 om ni får rate limits


def load_questions(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON måste vara en lista av objekt.")
    return data


def save_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    load_dotenv()

    client = OpenAIClient(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=MODEL,
        system_prompt=SYSTEM_PROMPT,
    )

    items = load_questions(INPUT_PATH)

    results = []
    for i, item in enumerate(items, start=1):
        qid = item.get("question_id")
        question = item.get("question", "")

        record = {
            "question_id": qid,
            "question": question,
            "expected_answer": item.get("expected_answer"),
            "source": item.get("source"),
            # valfritt: behåll chunk för analys, men baseline använder den inte
            "supporting_chunk": item.get("supporting_chunk"),
            "baseline_no_rag": {
                "model": MODEL,
                "system_prompt": SYSTEM_PROMPT,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "answer": None,
                "error": None,
            },
        }

        try:
            resp = client.invoke(question)
            record["baseline_no_rag"]["answer"] = resp.text
        except Exception as e:
            record["baseline_no_rag"]["error"] = f"{type(e).__name__}: {e}"

        results.append(record)

        if SLEEP_SECONDS_BETWEEN_CALLS > 0:
            time.sleep(SLEEP_SECONDS_BETWEEN_CALLS)

        print(f"[{i}/{len(items)}] {qid} -> {'OK' if record['baseline_no_rag']['error'] is None else 'ERROR'}")

    save_json(
        OUTPUT_PATH,
        {
            "run_info": {
                "run_type": "baseline_no_rag_batch",
                "model": MODEL,
                "system_prompt": SYSTEM_PROMPT,
                "input_path": INPUT_PATH,
                "created_utc": datetime.now(timezone.utc).isoformat(),
            },
            "results": results,
        },
    )
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()



'''
 query = "Vilket är det vanligaste färdmedlet bland de med en sammanlagd hushållsinkomst under 10 000 kronor per månad? "

load_dotenv()

client = OpenAIClient(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini",
    system_prompt="You are a helpful assistant that can answer questions and help with tasks.",
)

response = client.invoke(query)
print(response.text)
'''
