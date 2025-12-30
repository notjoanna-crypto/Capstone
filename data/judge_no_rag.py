import json
import os
from dotenv import load_dotenv
from openai import OpenAI

# data/judge_without_rag.py
# Run with: docker compose exec app python data/judge_without_rag.py

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

GROUND_TRUTH_FILE = "data/ground_truth_final.json"
BASELINE_FILE = "data/without_rag_answers.json"
OUTPUT_FILE = "data/judged_without_rag.json"

MODEL = "gpt-4o-mini"

JUDGE_PROMPT = """
You are an evaluation judge for a baseline LLM WITHOUT retrieval (no RAG).
Evaluate the model answer using ONLY the expected ground-truth answer.
Do NOT use outside knowledge.

Definitions:

Correctness:
- 1 if the model answer matches the expected ground-truth answer in meaning.
- 0 otherwise.

Hallucination:
- 1 if the model answer contains incorrect or fabricated factual information beyond what is supported by the expected answer.
- 0 if the answer is consistent with the expected answer (it may be shorter, but must not introduce wrong facts).

Question:
{question}

Expected ground-truth answer:
{expected_answer}

Model answer:
{generated_answer}

Return STRICTLY in JSON:
{{
  "correctness": 0,
  "hallucination": 0,
  "justification": {{
    "correctness": "...",
    "hallucination": "..."
  }}
}}
"""

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def safe_get_answer(baseline_record: dict) -> str:
    # Supports both of these output formats:
    # A) {"results":[{"question_id":..., "baseline_no_rag":{"answer":"..."}}]}
    # B) [{"question_id":..., "baseline_no_rag":{"answer":"..."}}]
    b = baseline_record.get("baseline_no_rag", {})
    return (b.get("answer") or "").strip()

def judge_one(question: str, expected_answer: str, generated_answer: str) -> dict:
    prompt = JUDGE_PROMPT.format(
        question=question,
        expected_answer=expected_answer,
        generated_answer=generated_answer,
    )

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    content = resp.choices[0].message.content.strip()
    return json.loads(content)

def main():
    ground_truth = load_json(GROUND_TRUTH_FILE)

    baseline = load_json(BASELINE_FILE)
    baseline_results = baseline["results"] if isinstance(baseline, dict) and "results" in baseline else baseline

    baseline_map = {r["question_id"]: r for r in baseline_results if "question_id" in r}

    judged = []
    for gt in ground_truth:
        qid = gt.get("question_id")
        if not qid or qid not in baseline_map:
            continue

        baseline_rec = baseline_map[qid]
        generated_answer = safe_get_answer(baseline_rec)

        out = {
            "question_id": qid,
            "question": gt.get("question"),
            "expected_answer": gt.get("expected_answer"),
            "generated_answer": generated_answer,
            "correctness": None,
            "hallucination": None,
            "justification": None,
            "error": None,
            "judge_model": MODEL,
        }

        try:
            evaluation = judge_one(
                question=gt.get("question", ""),
                expected_answer=gt.get("expected_answer", ""),
                generated_answer=generated_answer,
            )
            out["correctness"] = evaluation.get("correctness")
            out["hallucination"] = evaluation.get("hallucination")
            out["justification"] = evaluation.get("justification")
        except Exception as e:
            out["error"] = f"{type(e).__name__}: {e}"

        judged.append(out)
        print(f"{qid}: {'OK' if out['error'] is None else 'ERROR'}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_info": {
                    "run_type": "judge_without_rag",
                    "judge_model": MODEL,
                    "ground_truth_file": GROUND_TRUTH_FILE,
                    "baseline_file": BASELINE_FILE,
                },
                "results": judged,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nSaved judged results to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
