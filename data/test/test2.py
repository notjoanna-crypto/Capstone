#from datapizza.modules.parsers.docling import DoclingParser
#help(DoclingParser)
# Run code with: docker compose exec app python data/test2.py


from datapizza.modules.splitters import PDFImageSplitter

# Split while preserving images and layout
pdf_splitter = PDFImageSplitter()

pdf_chunks = pdf_splitter("data\resvaneundersokning_test.pdf")

# Examine chunks with visual content
for i, chunk in enumerate(pdf_chunks):
    print(f"Chunk {i+1}:")
    print(f"  Content length: {len(chunk.content)}")
    print(f"  Page: {chunk.metadata.get('page_number', 'unknown')}")

    if hasattr(chunk, 'media') and chunk.media:
        print(f"  Media elements: {len(chunk.media)}")
        for media in chunk.media:
            print(f"    Type: {media.media_type}")

    if 'boundingRegions' in chunk.metadata:
        print(f"  Bounding regions: {len(chunk.metadata['boundingRegions'])}")

    print("---")


doc = parser.parse("/app/resvaneundersokning.pdf")
# skriv ut första 3 paragraf-noderna du ser
count = 0
for sec in doc.children:
    for node in sec.children:
        if getattr(node, "content", None):
            print(node.content[:400])
            print("----")
            count += 1
            if count >= 3:
                return
