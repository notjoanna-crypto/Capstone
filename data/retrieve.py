import os
from dotenv import load_dotenv

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

# query = "Vad handlar resvaneundersökningen om?"

query = "Hur gamla är störst andel av de svarande?"
# query = "Vilket är det vanligaste färdmedlet bland de med en sammanlagd hushållsinkomst under 10 000 kronor per månad? "
#query = "How old are the people that answered ?"

result = dag_pipeline.run({
    "rewriter": {"user_prompt": query},
    "prompt": {"user_prompt": query},
    "retriever": {"collection_name": COLLECTION, "k": 3},
    "generator": {"input": query},
})

print(result["generator"])



# Run code with: docker compose exec app python data/retrieve.py

