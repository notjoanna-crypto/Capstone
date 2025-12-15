from pypdf import PdfReader, PdfWriter

reader = PdfReader("resvaneundersokning.pdf")
writer = PdfWriter()

# sidor är 0-indexerade → 0,1 = sida 1–2
for i in range(2):
    writer.add_page(reader.pages[i])

with open("data/resvaneundersokning_test.pdf", "wb") as f:
    writer.write(f)



# Run code with: docker compose exec app python data/makesmallerpdffile.py

