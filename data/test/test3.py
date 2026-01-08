from datapizza.modules.splitters import PDFImageSplitter
# Run code with: docker compose exec app python data/test/test3.py

# Split while preserving images and layout
pdf_splitter = PDFImageSplitter()

pdf_chunks = pdf_splitter("/app/documents/resvaneundersokning.pdf")


import fitz

pdf_path = "/app/documents/resvaneundersokning.pdf"

# 1) Image chunks (ger image_path)
splitter = PDFImageSplitter()
img_chunks = splitter(pdf_path)

# 2) Text per sida
doc = fitz.open(pdf_path)

for i, img_chunk in enumerate(img_chunks):
    page = doc[i]  # antar 1 chunk per sida i ordning
    text = page.get_text("text")
    image_path = img_chunk.metadata.get("image_path")

    print(f"Page {i+1}")
    print("image_path:", image_path)
    print("text_snippet:", text[:200])
    print("----")



print("--NY--")

page = doc[3]  # antar 1 chunk per sida i ordning
text = page.get_text("text")
print(text)