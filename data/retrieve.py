import os
from dotenv import load_dotenv

from datapizza.clients.openai import OpenAIClient
from datapizza.embedders.openai import OpenAIEmbedder
from datapizza.modules.prompt import ChatPromptTemplate
from datapizza.modules.rewriters import ToolRewriter
from datapizza.pipeline import DagPipeline
from datapizza.vectorstores.qdrant import QdrantVectorstore

load_dotenv()

COLLECTION = "my_documents"
VECTOR_NAME = "text-embedding-3-small"

openai_client = OpenAIClient(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
)

query_rewriter = ToolRewriter(
    client=openai_client,
    system_prompt="Rewrite the user query into a concise standalone search query. Do not answer.",
)

embedder = OpenAIEmbedder(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name=VECTOR_NAME,
)

# Connect to the SAME persistent Qdrant as ingestion
retriever = QdrantVectorstore(host="qdrant", port=6333)

prompt_template = ChatPromptTemplate(
    user_prompt_template="User question: {{user_prompt}}\n",
    retrieval_prompt_template="Retrieved content:\n{% for chunk in chunks %}{{ chunk.text }}\n{% endfor %}",
)

dag_pipeline = DagPipeline()
dag_pipeline.add_module("rewriter", query_rewriter)
dag_pipeline.add_module("embedder", embedder)
dag_pipeline.add_module("retriever", retriever)
dag_pipeline.add_module("prompt", prompt_template)
dag_pipeline.add_module("generator", openai_client)

dag_pipeline.connect("rewriter", "embedder", target_key="text")
dag_pipeline.connect("embedder", "retriever", target_key="query_vector")
dag_pipeline.connect("retriever", "prompt", target_key="chunks")
dag_pipeline.connect("prompt", "generator", target_key="memory")

query = "Vad handlar resvaneundersökningen om?"

result = dag_pipeline.run({
    "rewriter": {"user_prompt": query},
    "prompt": {"user_prompt": query},
    "retriever": {"collection_name": COLLECTION, "k": 3},
    "generator": {"input": query},
})

print(result["generator"])
