import os
from dotenv import load_dotenv

from datapizza.embedders import ChunkEmbedder
from datapizza.embedders.openai import OpenAIEmbedder
from datapizza.modules.parsers.docling import DoclingParser
from datapizza.modules.splitters import NodeSplitter
from datapizza.pipeline import IngestionPipeline
from datapizza.vectorstores.qdrant import QdrantVectorstore

load_dotenv()

COLLECTION = "resvaneundersokning_document"
VECTOR_NAME = "text-embedding-3-small"
DIM = 1536

CONFLICT_PDFS = [
    ("Conflict_Doc_1.pdf", 1),
    ("Conflict_Doc_2.pdf", 2),
    ("Conflict_Doc_3.pdf", 3),
    ("Conflict_Doc_4.pdf", 4),
    ("Conflict_Doc_5.pdf", 5),
]

def main():
    # 1) Connect to existing Qdrant
    vectorstore = QdrantVectorstore(host="qdrant", port=6333)

    # 2) Embedder (SAME as before)
    embedder = OpenAIEmbedder(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name=VECTOR_NAME,
    )

    # 3) Same ingestion pipeline
    ingestion_pipeline = IngestionPipeline(
        modules=[
            DoclingParser(),
            NodeSplitter(max_char=1000),
            ChunkEmbedder(client=embedder),
        ],
        vector_store=vectorstore,
        collection_name=COLLECTION,
    )

    # 4) Ingest ONLY conflict docs
    for pdf, cid in CONFLICT_PDFS:
        ingestion_pipeline.run(
            pdf,
            metadata={
                "source": "synthetic_conflict",
                "type": "pdf",
                "conflict_doc": True,
                "conflict_id": cid,
                "doc_id": pdf,
            },
        )

    print("Conflict ingestion complete.")

if __name__ == "__main__":
    main()
