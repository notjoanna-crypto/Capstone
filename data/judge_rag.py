import json
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

GROUND_TRUTH_FILE = "data/ground_truth_questions.json"

RESULTS_FILE = "data/results_clean_rag_k3.json"
OUTPUT_FILE = "data/judged_clean_rag.json" 

#RESULTS_FILE = "data/results_no_rag.json"
#OUTPUT_FILE = "data/judged_no_rag.json"

#RESULTS_FILE = "data/results_conflict_noise_rag.json"
#OUTPUT_FILE = "data/judged_conflict_noise_rag.json"


JUDGE_PROMPT = """
You are an evaluation judge for a Retrieval-Augmented Generation (RAG) system. 
Evaluate the model answer on THREE dimensions using ONLY the provided information.
Do NOT use outside knowledge.

Definitions:

Correctness:
- 1 if the model answer matches the expected ground-truth answer in both meaning and key details.
- 0 otherwise.

Hallucination:
- 1 if the model answer contains ANY factual claim not supported by the retrieved chunks.
- 0 if EVERY factual claim in the model answer is supported by at least one retrieved chunk.

Source Drift:
- Score 1 if the answer cites or relies on a document that is NOT the ground-truth source document (GT). 
- Score 0 if the answer is grounded in the ground-truth source document (GT).

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
  "justification": {{
    "correctness": "...",
    "hallucination": "...",
    "source_drift": "..."
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
            # model= "gemini-1.5-pro",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        evaluation = json.loads(response.choices[0].message.content)

        judged.append({
            "question_id": qid,
            "correctness": evaluation["correctness"],
            "hallucination": evaluation["hallucination"],
            "source_drift": evaluation["source_drift"],
            "justification": evaluation["justification"]
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(judged, f, indent=2, ensure_ascii=False)

    print(f"Saved judged results to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
    
    
# run code with: docker compose exec app python data/judge_rag.py
