

# Run code with: docker compose exec app python data/test/test4.py

from dotenv import load_dotenv
from pathlib import Path
from datapizza.clients.openai import OpenAIClient
from datapizza.modules.captioners import LLMCaptioner
from datapizza.type import Media, MediaNode, NodeType
import os
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


load_dotenv()

client = OpenAIClient(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o",
)



img_path = Path(__file__).parent / "page_5_0d829b28-1209-4cd1-b903-f0865ef68dc1.png"

captioner = LLMCaptioner(
    client=client,
    max_workers=3,
    system_prompt_figure="Convert this figure into a clear table. Use exact numbers if available, otherwise estimate based on bar lengths. Include all transport modes, categories, and comparisons shown in the figure."
)

document_node = MediaNode(
    node_type=NodeType.FIGURE,
    children=[],
    metadata={},
    media=Media(source_type="path", source=str(img_path), extension="png", media_type="image")
)

captioned_document = captioner(document_node)
print(captioned_document)


captioned_document = captioner(document_node)

print(captioned_document.metadata)

for child in captioned_document.children:
    print(child.metadata)
    if hasattr(child, "content"):
        print(child.content)
