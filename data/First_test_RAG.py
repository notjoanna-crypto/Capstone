from datapizza.clients.openai import OpenAIClient
from datapizza.core.vectorstore import VectorConfig
from datapizza.embedders import ChunkEmbedder
from datapizza.embedders.openai import OpenAIEmbedder
from datapizza.modules.captioners import LLMCaptioner
from datapizza.modules.parsers.docling import DoclingParser
from datapizza.modules.splitters import NodeSplitter
from datapizza.pipeline import IngestionPipeline
from datapizza.vectorstores.qdrant import QdrantVectorstore

# Run code with
# docker compose exec app python data/First_test_RAG.py


# 1. Create temporary vector storage  

vectorstore = QdrantVectorstore(location=":memory:")

# 2. Create a collection (dataset) "my_documents", which contains vectors of length 1536 (ie v=(x1​,x2​,…,x1536​))
# The vectors need to be of the same length, to be able to compare them.

vectorstore.create_collection(
    "my_documents",
    vector_config=[VectorConfig(name="embedding", dimensions=1536)]
)

# 3. Function which transformes text into vectors, this is done with openAI

embedder_client = OpenAIEmbedder(
    api_key="OPENAI_API_KEY",
    model_name="text-embedding-3-small",
)


########################################################################
####################### Ingestion Pipeline #############################
########################################################################


# In th ingestiov pipeline, documents are processed, split into chunks. 
# Embeddings are generated and stored in the vector database. 
