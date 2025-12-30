import os
from dotenv import load_dotenv
import json

from datapizza.clients.openai import OpenAIClient
from datapizza.embedders.openai import OpenAIEmbedder
from datapizza.modules.prompt import ChatPromptTemplate
from datapizza.modules.rewriters import ToolRewriter
from datapizza.pipeline import DagPipeline
from datapizza.vectorstores.qdrant import QdrantVectorstore

load_dotenv()

# 1. resvaneundersokning_document is the name of the collection containing vectors from the 90 page document
#    text-embedding-3-small is the name of the model. 

COLLECTION = "resvaneundersokning_document"
VECTOR_NAME = "text-embedding-3-small"


# 1. Create a client with the LLM gpt-4o-mini by loading the OpenAI API key from the environment.

openai_client = OpenAIClient(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
)

# 2. Rewite the user queries into better prompts.

query_rewriter = ToolRewriter(
    client=openai_client,
    system_prompt="Rewrite the user query into a concise standalone search query. Do not answer.",
)

# 3. This embedder will be used to transforms the user question/query into a vector and qdrant will compare them 
#    with the stored document vectors. The same embedder model "text-embedding-3-small" is used
#    in both cases to get the same length of vectors, making them comparable. 

embedder = OpenAIEmbedder(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name=VECTOR_NAME,
)

# 4. This is a vector client store and it creates a connection to the Qdrant, so it will talk to the Qdrant
#    and perform retrivals when a search/query method is called on it.

retriever = QdrantVectorstore(host="qdrant", port=6333)

# 5. This code defines how the final prompt sent to the LLM is constructed.
# It specifies a template for the user’s question and a template for the retrieved text chunks.
# The retrieved chunks are iterated over and their text is inserted into the prompt.
# The result is a combined prompt containing both the question and relevant context.
# This prompt is then passed to the LLM to generate an answer.

prompt_template = ChatPromptTemplate(
    user_prompt_template="User question: {{user_prompt}}\n",
    retrieval_prompt_template="Retrieved content:\n{% for chunk in chunks %}{{ chunk.text }}\n{% endfor %}",
)

# 6. This creates a DagPipeline object, which is a container for the pipeline.
dag_pipeline = DagPipeline()
dag_pipeline.add_module("rewriter", query_rewriter)

# 6.1 This line adds the embedder object to the DAG pipeline under the name "embedder".
# It registers the embedder as a pipeline step that can be executed.
# Other modules can now depend on or receive output from this embedder.
dag_pipeline.add_module("embedder", embedder)

# 6.2  This registers the QdrantVectorstore as a retriever module in the pipeline.
dag_pipeline.add_module("retriever", retriever)

# 6.3 This line adds the prompt template to the DAG pipeline under the name "prompt".
#  registers a step responsible for constructing the final prompt text.
dag_pipeline.add_module("prompt", prompt_template)

# 6.4 This line adds the LLM generator to the DAG pipeline under the name "generator".
dag_pipeline.add_module("generator", openai_client)

dag_pipeline.connect("rewriter", "embedder", target_key="text")


# 7. This line connects the embedder’s output to the retriever’s input in the pipeline.
dag_pipeline.connect("embedder", "retriever", target_key="query_vector")

# 8. This wires the output of the retriever to the input of the prompt module
dag_pipeline.connect("retriever", "prompt", target_key="chunks")

# 9. This line connects the prompt module’s output to the generator (LLM) in the pipeline.
dag_pipeline.connect("prompt", "generator", target_key="memory")


""" Examples of queries to test the RAG system."""
"""""""""
# query = "What are the seven main categories of transport modes used in the survey, and which specific modes are included in each category?"
query = "What is the target market share for public transport compared to the current market share mentioned in the text?"


# 11. This call executes the full DAG pipeline once, and it returns k chunks. 
result = dag_pipeline.run({
    "rewriter": {"user_prompt": query},
    "prompt": {"user_prompt": query},
    "retriever": {"collection_name": COLLECTION, "k": 3},
    "generator": {"input": query},
})

print("-" * 101 + "\n" +"-" * 41 + " Answer to prompt: " + "-" * 41 + "\n" + "-" * 101)
print(result["generator"].text)


# 12. This code retrives top 3 chunks and from where they were obtained. 

print("-" * 101 + "\n" +"-" * 40 + " Source and results: " + "-" * 40 + "\n" + "-" * 101)

retrieved_chunks = result["retriever"]

for chunk in retrieved_chunks:
    text = chunk.text
    meta = chunk.metadata

    source = meta.get("source")
    page = meta.get("page_no")
    
    print(f"Source: {source}")
    print(f"Page: {page}")
    print("Text:")
    print(text)
    print("-" * 101)

"""""""""
# 10. Prompts/ Queries from the ground truth file are loaded and run through the RAG system to get answers.

# input ground truth file with questions
GT_FILE = "data/ground_truth_final.json"
# output file to save results
OUTPUT_FILE = "data/results_clean_rag.json"
# Load ground truth questions
with open(GT_FILE, "r", encoding="utf-8") as f:
    ground_truth = json.load(f)

# Run each question through the pipeline and collect results
results = []

for item in ground_truth:
    question_id = item["question_id"]
    question = item["question"]
    expected_answer = item["expected_answer"]

    result = dag_pipeline.run({
        "rewriter": {"user_prompt": question},
        "prompt": {"user_prompt": question},
        "retriever": {"collection_name": COLLECTION, "k": 3},
        "generator": {"input": question},
    })

    generated_answer = result["generator"].text
    retrieved_chunks = result["retriever"]

    chunks_out = []
    for chunk in retrieved_chunks:
        chunks_out.append({
            "page": chunk.metadata.get("page_no"),
            "source": chunk.metadata.get("source"),
            "text": chunk.text
        })

    results.append({
        "question_id": question_id,
        "question": question,
        "expected_answer": expected_answer,
        "generated_answer": generated_answer,
        "retrieved_chunks": chunks_out,
    })
    
# Save results to output file
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Saved Clean RAG results to {OUTPUT_FILE}")


# Run code with: docker compose exec app python data/retrieve.py

