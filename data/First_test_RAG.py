from datapizza.clients.openai import OpenAIClient
from datapizza.core.vectorstore import VectorConfig
from datapizza.embedders import ChunkEmbedder
from datapizza.embedders.openai import OpenAIEmbedder
from datapizza.modules.captioners import LLMCaptioner
from datapizza.modules.parsers.docling import DoclingParser
from datapizza.modules.splitters import NodeSplitter
from datapizza.pipeline import IngestionPipeline
from datapizza.vectorstores.qdrant import QdrantVectorstore

from datapizza.modules.prompt import ChatPromptTemplate
from datapizza.modules.rewriters import ToolRewriter
from datapizza.pipeline import DagPipeline

import os
from dotenv import load_dotenv

load_dotenv()

# Run code with:
# docker compose exec app python data/First_test_RAG.py


# 1. Create temporary vector storage  

vectorstore = QdrantVectorstore(location=":memory:")

# 2. Create a collection (dataset) "my_documents", which contains vectors of length 1536 (ie v=(x1​,x2​,…,x1536​))
# The vectors need to be of the same length, to be able to compare them.

vectorstore.create_collection(
    "my_documents",
    vector_config=[VectorConfig(name="embedding", dimensions=1536)]
)

# 3. Function which transformes text into vectors, this is done with openAI.
#    OpenAIEmbedder provides text to numbers

embedder_client = OpenAIEmbedder(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small",
)


########################################################################
####################### Ingestion Pipeline #############################
########################################################################


# 1. In the ingestion pipeline, documents are processed and split into chunks. 
# Embeddings are generated and stored in the vector database. 

# DoclingParser, reads the documents and extracts text,
# Nodesplitter divides the text into smaller pieces (max 1000 char)
# ChunkEmbedder transforms each chunk of text into a list of 1536 number,
# where the same text has similar number,
# Then save it in the collection my_documents inside the vectorstore   

ingestion_pipeline = IngestionPipeline(
    modules=[
        DoclingParser(), # choose between Docling, Azure or TextParser to parse plain text

        #LLMCaptioner(
        #    client=OpenAIClient(api_key="YOUR_API_KEY"),
        #), # This is optional, add it if you want to caption the media

        NodeSplitter(max_char=1000),             # Split Nodes into Chunks
        ChunkEmbedder(client=embedder_client),   # Add embeddings to Chunks
    ],
    vector_store=vectorstore,
    collection_name="my_documents"
)


# 2. Run the ingestion pipeline on Resvaneundersökningen 2023
#    and store it with source name "resvaneundersokning"

ingestion_pipeline.run("resvaneundersokning.pdf", metadata={"source": "resvaneundersokning", "type": "pdf"})


# 3. Search for the most similar vectors in the collection, and compare it
#    to a vector with only zeros, it will be used with real queries later. 
#    Return the 2 vectors which are closest to the null vector.

res = vectorstore.search(
    query_vector = [0.0] * 1536,
    collection_name="my_documents",
    k=2,
)

print(res)



########################################################################
################### Retrieval with DagPipeline #########################
########################################################################


# 1. Create the LLM which is to answer the questions, the gpt-4o-mini model is fast and  cheap
#    OpenAIClient provides text to text
openai_client = OpenAIClient(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)

