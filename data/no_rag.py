import os
import json
import re
from dotenv import load_dotenv
from pypdf import PdfReader

from datapizza.clients.openai import OpenAIClient
from datapizza.modules.prompt import ChatPromptTemplate
from datapizza.pipeline import DagPipeline

load_dotenv()

# CONFIG
PDF_FILE = "resvaneundersokning.pdf"
GT_FILE = "data/ground_truth_questions.json"
OUTPUT_FILE = "data/results_no_rag.json"

MODEL_NAME = "gpt-4o-mini"

if not os.path.exists(PDF_FILE):
    raise FileNotFoundError(f"PDF not found at: {PDF_FILE}")

# LOAD PDF 
def load_pdf_with_pages(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    pages = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages.append(f"[PAGE {i + 1}]\n{text}")

    return "\n\n".join(pages)

DOCUMENT_TEXT = load_pdf_with_pages(PDF_FILE)

# FAKE CHUNK (Datapizza-compatible)
# Datapizza's ChatPromptTemplate always expects an iterable of "chunks"
# with a `.text` attribute. In the no-RAG baseline, no retrieval is
# performed, so the entire document is wrapped in a single pseudo-chunk.
# This preserves the same prompt structure as the RAG pipeline while
# explicitly disabling retrieval and metadata grounding.
class FakeChunk:
    def __init__(self, text):
        self.text = text

DOCUMENT_CHUNK = FakeChunk(DOCUMENT_TEXT)

# PAGE EXTRACTION
def extract_pages(answer_text: str):
    pages = set()

    # 1. Extract explicit "Pages: X"
    match = re.search(r"Pages:\s*([0-9,\s]+)", answer_text)
    if match:
        return match.group(1).strip()

    # 2. Fallback: extract [PAGE X] markers
    for p in re.findall(r"\[PAGE\s+(\d+)\]", answer_text):
        pages.add(p)

    if pages:
        return ", ".join(sorted(pages, key=int))

    return None

# LLM CLIENT
openai_client = OpenAIClient(
    model=MODEL_NAME,
    api_key=os.getenv("OPENAI_API_KEY"),
)

# PROMPT TEMPLATE (NO-RAG)
prompt_template = ChatPromptTemplate(
    user_prompt_template="""
You are an assistant that answers questions strictly based on the provided document.

Rules:
- Use only the document.
- Cite the page number(s) where the answer is found.
- If the answer is not stated, reply exactly:
"Not stated in the document."

Question:
{{user_prompt}}

Answer format:
Answer: <answer text>
Pages: <page numbers>
""",
    retrieval_prompt_template="""
Document:
{% for chunk in chunks %}
{{ chunk.text }}
{% endfor %}
"""
)

# PIPELINE
dag_pipeline = DagPipeline()
dag_pipeline.add_module("prompt", prompt_template)
dag_pipeline.add_module("generator", openai_client)

dag_pipeline.connect("prompt", "generator", target_key="memory")


# LOAD QUESTIONS
with open(GT_FILE, "r", encoding="utf-8") as f:
    ground_truth = json.load(f)

# RUN EVALUATION
results = []

for item in ground_truth:
    question_id = item["question_id"]
    question = item["question"]
    expected_answer = item["expected_answer"]

    result = dag_pipeline.run({
        "prompt": {
            "user_prompt": question,
            "chunks": [DOCUMENT_CHUNK]  # full PDF injected
        },
        "generator": {
            "input": question
        }
    })

    generated_answer = result["generator"].text
    pages = extract_pages(generated_answer)

    results.append({
        "question_id": question_id,
        "question": question,
        "expected_answer": expected_answer,
        "generated_answer": generated_answer,
        "retrieved_chunks": [
            {
                "source": "GT.pdf",
                "page": pages if pages else "model-cited"
            }
        ]
    })

# SAVE RESULTS
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Saved No_RAG results to {OUTPUT_FILE}")


# run: docker compose exec app python data/no_rag.py