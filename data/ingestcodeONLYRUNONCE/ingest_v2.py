# docker compose exec app python data/ingestcodeONLYRUNONCE/ingest_v2.py

import os
from dotenv import load_dotenv

from datapizza.core.vectorstore import VectorConfig
from datapizza.embedders import ChunkEmbedder
from datapizza.embedders.openai import OpenAIEmbedder
from datapizza.modules.parsers.docling import DoclingParser
from datapizza.modules.splitters import NodeSplitter
from datapizza.pipeline import IngestionPipeline
from datapizza.vectorstores.qdrant import QdrantVectorstore
from datapizza.modules.parsers.docling.ocr_options import OCROptions, OCREngine
from datapizza.modules.captioners import LLMCaptioner
from datapizza.clients.openai import OpenAIClient
load_dotenv()


# COLLECTION = "my_documents"
COLLECTION = "resvaneundersokning_ComponentChange_Conflict" 
VECTOR_NAME = "text-embedding-3-small"
DIM = 1536
PDF_PATH = "/app/documents/resvaneundersokning.pdf"


# 1) Connect to persistent Qdrant (docker compose service name)
vectorstore = QdrantVectorstore(host="qdrant", port=6333)
    
# 2) Create collection needed (same vector name as retrieval)
vectorstore.create_collection(
    COLLECTION,
    vector_config=[VectorConfig(name=VECTOR_NAME, dimensions=DIM)],
)

# 3) OCR 

parser = DoclingParser(
    ocr_options=OCROptions(engine=OCREngine.NONE)
)


'''
parser = DoclingParser(
    ocr_options=OCROptions(
        engine=OCREngine.TESSERACT,
        tesseract_lang=["swe"],
    )
)
'''

# 4. Client
client = OpenAIClient(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o",
)

# 4. Captioner for better tables
captioner = LLMCaptioner(
    client=client,
    system_prompt_figure="Convert this figure into a clear table. Use exact numbers if available, otherwise estimate based on bar lengths. Include all transport modes, categories, and comparisons shown in the figure."
)

# 3) Embedder (text -> vector)
embedder = OpenAIEmbedder(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name=VECTOR_NAME,
)


ingestion_pipeline = IngestionPipeline(
    modules=[
        parser,
        captioner,
        NodeSplitter(max_char=1000),
        ChunkEmbedder(client=embedder),
    ],
    vector_store=vectorstore,
    collection_name=COLLECTION,
)


ingestion_pipeline.run(
    PDF_PATH,
    metadata={"source": "resvaneundersokning", "type": "pdf"},
)

print("Ingestion complete.")