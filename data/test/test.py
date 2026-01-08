from qdrant_client import QdrantClient
COLLECTION = "resvaneundersokning_document_v2"


qdrant_client = QdrantClient(host="qdrant", port=6333)

count = qdrant_client.count(
    collection_name=COLLECTION,
    exact=True
)
print("Total chunks:", count.count)
# Run code with: docker compose exec app python data/test/test.py


from qdrant_client import QdrantClient

COLLECTION = "resvaneundersokning_ComponentChange_Conflict"
qdrant = QdrantClient(host="qdrant", port=6333)

lengths = []
offset = None

while True:
    points, offset = qdrant.scroll(
        collection_name=COLLECTION,
        limit=256,
        offset=offset,
        with_payload=True,
        with_vectors=False,
    )
    for p in points:
        payload = p.payload or {}
        text = payload.get("text") or payload.get("content") or payload.get("chunk")
        if text:
            lengths.append(len(text))
    if offset is None:
        break

print("Chunks with text:", len(lengths))
print("Min length:", min(lengths))
print("Max length:", max(lengths))
print("Average length:", sum(lengths) / len(lengths))
print("Below 100 chars:", sum(l < 20 for l in lengths))
print("Below 200 chars:", sum(l > 200 and l < 500 for l in lengths))


from qdrant_client import QdrantClient

qdrant = QdrantClient(host="qdrant", port=6333)

offset = None

while True:
    points, offset = qdrant.scroll(
        collection_name=COLLECTION,
        limit=128,
        offset=offset,
        with_payload=True,
        with_vectors=False,
    )

    for p in points:
        payload = p.payload or {}
        text = payload.get("text") or payload.get("content") or payload.get("chunk")

        if text and len(text) >= 200 and len(text) <= 220:
            print("=" * 80)
            print(f"Length: {len(text)}")
            print(text)

    if offset is None:
        break



import re
from qdrant_client import QdrantClient

qdrant = QdrantClient(host="qdrant", port=6333)

FIG_TABLE_PAT = re.compile(r"\b(figur|tabell|diagram)\b", re.IGNORECASE)
MANY_NUM_PAT = re.compile(r"\d")
PCT_PAT = re.compile(r"%")
BULLET_PAT = re.compile(r"^\s*([-•*]|(\d+[\.\)]))\s+", re.MULTILINE)

def is_running_text(text: str) -> bool:
    t = text.strip()
    if len(t) < 200:  # justera efter dina chunks
        return False

    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if not lines:
        return False

    # --- features ---
    n_chars = len(t)
    n_lines = len(lines)

    avg_line_len = sum(len(ln) for ln in lines) / n_lines
    short_lines = sum(1 for ln in lines if len(ln) < 25)
    short_line_ratio = short_lines / n_lines

    digit_ratio = sum(1 for ch in t if ch.isdigit()) / n_chars
    pct_count = len(PCT_PAT.findall(t))

    newline_density = n_lines / max(1, n_chars / 80)  # ungefär "rader per 80 tecken"
    has_bullets = bool(BULLET_PAT.search(t))
    has_fig_table = bool(FIG_TABLE_PAT.search(t))

    # --- rules tuned for "brödtext" ---
    # 1) inte för "radig"
    if short_line_ratio > 0.35:
        return False
    if avg_line_len < 40:
        return False
    if newline_density > 2.2:
        return False

    # 2) inte för siffer-tung
    if digit_ratio > 0.12:
        return False
    if pct_count > 6:
        return False

    # 3) lista/tabell/figur-indikationer
    if has_bullets:
        return False
    # Om du vill exkludera allt som nämner Figur/Tabell:
    if has_fig_table:
        return False

    # 4) enkel mening-check: minst några skiljetecken som tyder på löpande text
    if (t.count(".") + t.count(",") + t.count(";") + t.count(":")) < 6:
        return False

    return True

offset = None
hits = 0

while True:
    points, offset = qdrant.scroll(
        collection_name=COLLECTION,
        limit=256,
        offset=offset,
        with_payload=True,
        with_vectors=False,
    )

    for p in points:
        payload = p.payload or {}
        text = payload.get("text") or payload.get("content") or payload.get("chunk")
        if text and is_running_text(text):
            hits += 1
            print("=" * 80)
            print(f"Running text hit #{hits} | length={len(text)} | id={p.id}")
            print(text[:1200])  # undvik spam

    if offset is None:
        break

print("Total running-text chunks:", hits)
