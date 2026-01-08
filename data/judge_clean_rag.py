import json
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

GROUND_TRUTH_FILE = "data/jsonFiler/QA_triplets.json"
RESULTS_FILE = "data/jsonFiler/results_conflict_k20.json"
OUTPUT_FILE = "data/jsonFiler/judged_conflict_k20.json"


JUDGE_PROMPT = """
You are an evaluation judge for a Retrieval-Augmented Generation (RAG) system. 
Evaluate the model answer on THREE dimensions using ONLY the provided information.
Do NOT use outside knowledge.

Definitions:

Correctness:
- 1 if the model answer matches the expected ground-truth answer in meaning.
- 0 otherwise.

Hallucination:
- 1 if the model answer contains ANY factual claim not supported by the retrieved chunks.
- 0 if ALL factual claims are supported by the retrieved chunks.

Source Drift:
- 1 if the model answer relies on or cites information from a document other than the ground-truth document.
- 0 if it relies only on the ground-truth document.

Correct Chunks:
- 1 only if the retrieved chunks explicitly contain the specific and decisive information required to answer the question with exactly the same meaning as the ground truth. All necessary facts must be directly stated in the chunks without assumptions or inference.
- 0 if any part of the required information is missing, implicit, or cannot be directly verified from the retrieved chunks.


Question:
{question}

Expected ground-truth answer:
{expected_answer}

Model answer:
{generated_answer}

Retrieved chunks:
{retrieved_chunks}

Return STRICTLY in JSON:
{{
  "correctness": 0 | 1,
  "hallucination": 0 | 1,
  "source_drift": 0 | 1,
  "correct_chunks": 0 | 1,
  "justification": {{
    "correctness": "...",
    "hallucination": "...",
    "source_drift": "...",
    "correct_chunks": "..."

  }}
}}
"""

def main():
    with open(GROUND_TRUTH_FILE, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)
        
    # Map results by question_id
    results_map = {r["question_id"]: r for r in results}
    judged = []

    for gt in ground_truth:
        qid = gt["question_id"]
        if qid not in results_map:
            continue

        r = results_map[qid]

        prompt = JUDGE_PROMPT.format(
            question=gt["question"],
            expected_answer=gt["expected_answer"],
            generated_answer=r["generated_answer"],
            retrieved_chunks=json.dumps(r["retrieved_chunks"], indent=2, ensure_ascii=False)
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        evaluation = json.loads(response.choices[0].message.content)

        judged.append({
            "question_id": qid,
            "correctness": evaluation["correctness"],
            "hallucination": evaluation["hallucination"],
            "source_drift": evaluation["source_drift"],
            "correct_chunks": evaluation["correct_chunks"],
            "justification": evaluation["justification"]
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(judged, f, indent=2, ensure_ascii=False)

    print(f"Saved judged results to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
    
    
# run code with: docker compose exec app python data/judge_clean_rag.py
