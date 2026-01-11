import json
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

GROUND_TRUTH_FILE = "data/ground_truth_questions.json"

# Choose ONE results file to evaluate:

# Clean RAG variants
RESULTS_FILE = "data/final_results_jsonFiles/results_clean_rag_k3_v1.json"   # k=3
# RESULTS_FILE = "data/results_clean_rag_k3_v2.json"   # k=3
# RESULTS_FILE = "data/results_clean_rag_k5.json"   # k=5
# RESULTS_FILE = "data/results_clean_rag_k10.json"  # k=10
# RESULTS_FILE = "data/results_clean_rag_k15.json"  # k=15
# RESULTS_FILE = "data/results_clean_rag_k20.json"  # k=20

# Noise+Conflict RAG variants
# RESULTS_FILE = "data/results_conflict_noise_rag_k3_v1.json"   # k=3
# RESULTS_FILE = "data/results_conflict_noise_rag_k3_v2.json"   # k=3
# RESULTS_FILE = "data/results_conflict_noise_rag_k5.json"   # k=5
# RESULTS_FILE = "data/results_conflict_noise_rag_k10.json"  # k=10
#RESULTS_FILE = "data/results_conflict_noise_rag_k15.json"  # k=15
# RESULTS_FILE = "data/results_conflict_noise_rag_k20.json"  # k=20

# Output file (will contain hit rate and detailed judgments)
OUTPUT_FILE = "data/hit_rate_results.json"

# hit rate judgment prompt
HIT_RATE_PROMPT = """
You are judging whether a retrieved document chunk is relevant for answering a specific question.

Question: 
{question}

Expected Answer: 
{expected_answer}

Retrieved Chunk Information:
- Source Document: {source}
- Page: {page}
- Text: {chunk_text}

Judgment Criteria: A chunk is RELEVANT only if ALL THREE conditions are met:
1. Correct Document: Chunk must be from the GROUND TRUTH document (GT=resvaneundersokning)
2. Correct Page: Chunk must be from correct page (as specified in ground truth)
3. Contains Answer: Chunk must contain the expected answer or information needed to derive it

Instructions:
- Answer ONLY with "YES" or "NO" 
- YES = All three conditions are met
- NO = Any condition fails

Example Judgments:
- GT document, page 3, contains "49 percent" when answer is "49 percent" → YES
- GT document, page 3, doesn't contain the expected answer → NO
- Wrong document (Noise/Conflict), even if contains answer → NO
- Ground truth document, wrong page, contains answer → NO
"""

def judge_single_chunk(question, expected_answer, chunk, correct_pages):
    """Simple LLM judgment: YES if chunk is relevant, NO if not"""
    prompt = HIT_RATE_PROMPT.format(
        question=question,
        expected_answer=expected_answer,
        source=chunk['source'],
        page=chunk['page'],
        chunk_text=chunk['text'],   
        correct_pages=correct_pages
    )
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=5
    )
    
    answer = response.choices[0].message.content.strip().upper()
    return answer == "YES"

def main():
    # Load data
    with open(GROUND_TRUTH_FILE, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)
    
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)
    
    # Map results by question_id
    results_map = {r["question_id"]: r for r in results}
    
    # Initialize counters
    hits = 0
    total_questions = 0
    all_judgments = []
    
    print(f"Computing Hit Rate for: {RESULTS_FILE}")
    print(f"Ground truth: {GROUND_TRUTH_FILE}")
    print("-" * 60)
    
    # Process each question
    for gt in ground_truth:
        qid = gt["question_id"]
        
        if qid not in results_map:
            print(f"Warning: Question {qid} not found in results, skipping...")
            continue
        
        result = results_map[qid]
        question = gt["question"]
        expected_answer = gt["expected_answer"]
        correct_pages = gt["source"]["pages"]
        
        found_relevant = False
        question_judgments = []
        
        # Check each retrieved chunk for this question
        for chunk_idx, chunk in enumerate(result["retrieved_chunks"]):
            # Use LLM to judge relevance
            is_relevant = judge_single_chunk(
                question=question,
                expected_answer=expected_answer,
                chunk=chunk,
                correct_pages=correct_pages
            )
            
            # Record judgment for debugging
            question_judgments.append({
                "chunk_idx": chunk_idx,
                "source": chunk["source"],
                "page": chunk["page"],
                "text_preview": chunk["text"][:100] + "..." if len(chunk["text"]) > 100 else chunk["text"],
                "is_relevant": is_relevant
            })
            
            if is_relevant:
                found_relevant = True
                # No need to check remaining chunks for this question
                break # Found one relevant chunk, we're done!!
        
        # Update counters
        if found_relevant:
            hits += 1
        total_questions += 1
        
        # Store judgments for this question
        all_judgments.append({
            "question_id": qid,
            "question": question,
            "expected_answer": expected_answer,
            "found_relevant": found_relevant,
            "chunk_judgments": question_judgments
        })
        
        # Print progress
        print(f"Q{qid:3}: {'✓ HIT' if found_relevant else '✗ MISS'} | "
              f"Progress: {total_questions}/{len(ground_truth)}")
    
    # Calculate hit rate
    hit_rate = hits / total_questions if total_questions > 0 else 0
    
    print("\n" + "=" * 60)
    print(f"RESULTS SUMMARY")
    print("=" * 60)
    print(f"Results file: {RESULTS_FILE}")
    print(f"Total questions evaluated: {total_questions}")
    print(f"Questions with at least one relevant chunk: {hits}")
    print(f"Hit Rate: {hit_rate:.1%} ({hits}/{total_questions})")
    
    
    # Extract k value from filename 
    import re
    match = re.search(r'k(\d+)', RESULTS_FILE)
    k_value = match.group(1) if match else "N/A"
    print(f"Retrieval depth (k): {k_value}")
    
    
    # Prepare final results
    final_results = {
        "results_file": RESULTS_FILE,
        "ground_truth_file": GROUND_TRUTH_FILE,
        "hit_rate": hit_rate,
        "hits": hits,
        "total_questions": total_questions,
        "k": k_value,
        "judgments": all_judgments
    }
    
    # Save results
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed results saved to: {OUTPUT_FILE}")
    
    # Also save a simple summary for easy reading
    summary = {
        "config": {
            "results_file": os.path.basename(RESULTS_FILE),
            "k": k_value,
            "model": "gpt-4o-mini"
        },
        "metrics": {
            "hit_rate": hit_rate,
            "hits": hits,
            "total": total_questions
        }
    }
    
    summary_file = "data/hit_rate_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"Summary saved to: {summary_file}")
    
    return hit_rate

if __name__ == "__main__":
    main()
    
    
# run: docker compose exec app python data/hit_rate.py