import os
import json
import random
from dotenv import load_dotenv

from openai import OpenAI
from datapizza.vectorstores.qdrant import QdrantVectorstore
from datapizza.embedders.openai import OpenAIEmbedder



load_dotenv()

# CONFIG
GT_CHUNKS_FILE = "data/gt_chunks.json"

OUTPUT_FILE = "data/ground_truth_questions.json"

# CLIENTS
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
embedder = OpenAIEmbedder(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small",
)

# PROMPT
GROUND_TRUTH_PROMPT = """
You are generating ground-truth evaluation data for a Retrieval-Augmented Generation (RAG) system.

Your task:
1. Identify ALL distinct, self-contained factual claims stated in the text.
   - A factual claim is a statement that can be verified as true or false using the text alone
   - Numerical values, comparisons, trends, or categorical statements all count as separate claims
2. For EACH factual claim:
   a) Write ONE clear, precise factual question in ENGLISH that targets only that claim
   b) Write the exact correct answer in ENGLISH, based strictly on the text

Rules:
- Generate ONE question per factual claim (do NOT paraphrase the same claim)
- Do NOT invent claims that are not explicitly stated
- Do NOT use outside knowledge
- Do NOT infer beyond the text
- If the text contains only one factual claim, output only one question

Output strictly in JSON as a list:
[
  {{
    "question": "...",
    "expected_answer": "..."
  }}
]

DOCUMENT CHUNK:
---------
{chunk}
---------
"""


# MAIN
def main():
    print("[INFO] Starting ground-truth generation...")

    with open(GT_CHUNKS_FILE, "r", encoding="utf-8") as f:
        gt_chunks = json.load(f)

    ground_truth = []
    q_counter = 1

    for chunk in gt_chunks:
        chunk_text = chunk["text"]
        pages = chunk["pages"]
        chunk_id = chunk["chunk_id"]

        prompt = GROUND_TRUTH_PROMPT.format(chunk=chunk_text)

        completion = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        qa_list = json.loads(completion.choices[0].message.content)

        
        if len(qa_list) == 0:
            raise ValueError(f"No questions generated for {chunk_id}")


        for qa in qa_list:
            ground_truth.append({
                "question_id": f"Q_{q_counter:03}",
                "chunk_id": chunk_id,
                "question": qa["question"],
                "expected_answer": qa["expected_answer"],
                "source": {
                    "document": chunk["document"],
                    "pages": pages
                },
                "supporting_chunk": {
                    "chunk_id": chunk_id,
                    "text": chunk_text
                }
            })
            q_counter += 1

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Generated {len(ground_truth)} questions from {len(gt_chunks)} chunks.")
    print(f"[INFO] Saved to {OUTPUT_FILE}!!!")


if __name__ == "__main__":
    main()

# docker compose exec app python data/generate_ground_truth.py