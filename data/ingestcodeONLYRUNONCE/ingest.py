# docker compose exec app python data/ingest.py


import os
from dotenv import load_dotenv

from datapizza.core.vectorstore import VectorConfig
from datapizza.embedders import ChunkEmbedder
from datapizza.embedders.openai import OpenAIEmbedder
from datapizza.modules.parsers.docling import DoclingParser
from datapizza.modules.splitters import NodeSplitter
from datapizza.pipeline import IngestionPipeline
from datapizza.vectorstores.qdrant import QdrantVectorstore

load_dotenv()


# COLLECTION = "my_documents"
COLLECTION = "resvaneundersokning_document" 
VECTOR_NAME = "text-embedding-3-small"
DIM = 1536

PDF_PATH = "resvaneundersokning.pdf"

def main():
    # 1) Connect to persistent Qdrant (docker compose service name)
    vectorstore = QdrantVectorstore(host="qdrant", port=6333)

    # 2) Create collection needed (same vector name as retrieval)
    vectorstore.create_collection(
        COLLECTION,
        vector_config=[VectorConfig(name=VECTOR_NAME, dimensions=DIM)],
    )

    # 3) Embedder (text -> vector)
    embedder = OpenAIEmbedder(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name=VECTOR_NAME,
    )

    # 4) Ingestion pipeline
    ingestion_pipeline = IngestionPipeline(
        modules=[
            DoclingParser(),
            NodeSplitter(max_char=1000),
            ChunkEmbedder(client=embedder),
        ],
        vector_store=vectorstore,
        collection_name=COLLECTION,
    )

    # 5) Run ingestion once (or whenever docs change)
    ingestion_pipeline.run(
        PDF_PATH,
        metadata={"source": "resvaneundersokning", "type": "pdf"},
    )

    print("Ingestion complete.")

if __name__ == "__main__":
    main()
