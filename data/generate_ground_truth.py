import os
import json
import random
from dotenv import load_dotenv

from openai import OpenAI
from datapizza.vectorstores.qdrant import QdrantVectorstore
from datapizza.embedders.openai import OpenAIEmbedder



load_dotenv()

# CONFIG
COLLECTION = "resvaneundersokning_document"
OUTPUT_FILE = "ground_truth.json"
NUM_QUESTIONS = 50  # Number of ground-truth questions to generate

# CLIENTS
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
embedder = OpenAIEmbedder(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small",
)


retriever = QdrantVectorstore(host="qdrant", port=6333)

# PROMPT
GROUND_TRUTH_PROMPT = """
You are generating ground-truth evaluation data for a Retrieval-Augmented Generation (RAG) system.


DOCUMENT CHUNK:
---------
{chunk}
---------

Your task:
1. Write ONE clear, non-trivial factual question in ENGLISH that can be answered using ONLY the information in the text.
2. The question must require understanding the information in the text, not just reading a single number or title.
3. Write the exact correct answer in ENGLISH, based strictly on the text.

Constraints:
- Prefer questions involving comparisons, trends, proportions, or explanations stated in the text.
- Do NOT use outside knowledge.
- Do NOT infer beyond the text.

Output strictly in JSON:
{{
  "question": "...",
  "expected_answer": "..."
}}
"""

# MAIN
def main():
    # 1. Create a neutral embedding
    dummy_vector = embedder.embed(" ")
    
    # 2. Similarity search to fetch all chunks    
    points = retriever.search(
        query_vector=dummy_vector,
        collection_name=COLLECTION,
        k=2000  # Fetch a large number to sample from
    )

    # Inspect retrieved chunks
    print(f"INFO] Retrieved {len(points)} chunks from Qdrant for sampling.")
    example_point = points[0]
    print("[INFO] Example metadata keys:", example_point.metadata.keys())
    print("[INFO] Page:", example_point.metadata.get("page_no"))
    print("[INFO] Text preview:", example_point.text[:200])

    if len(points) < NUM_QUESTIONS:
        raise ValueError("Not enough chunks to sample from.")

    # 2. Uniform sampling across retrieved chunks
    step = max(len(points) // NUM_QUESTIONS, 1)
    sampled = points[::step][:NUM_QUESTIONS]

    ground_truth = []

    for i, point in enumerate(sampled, start=1):
       
        chunk_text = point.text
        page = point.metadata.get("page_no")
        prompt = GROUND_TRUTH_PROMPT.format(chunk=chunk_text)
        completion = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        qa = json.loads(completion.choices[0].message.content)


        ground_truth.append({
            "question_id": f"GT_{i:03}",
            "question": qa["question"],
            "expected_answer": qa["expected_answer"],
            "source": {
                "document": "resvaneundersokning.pdf",
                "pages": [page]
            },
            "supporting_chunk": {
                "page": page,
                "text": chunk_text
            }
        })
        
    print(f"[INFO] Sampling {len(sampled)} chunks for ground truth.")


    # 3. Save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2, ensure_ascii=False)

    print(f"Generated {NUM_QUESTIONS} ground-truth questions saved to {OUTPUT_FILE}!!")
    


if __name__ == "__main__":
    main()


# Run code with: docker compose exec app python data/generate_ground_truth.py


