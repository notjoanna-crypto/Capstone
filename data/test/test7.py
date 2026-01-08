from qdrant_client import QdrantClient

c = QdrantClient(host="qdrant", port=6333)
COLLECTION = "resvaneundersokning_ComponentChange"

points, _ = c.scroll(collection_name=COLLECTION, with_payload=True, with_vectors=False, limit=2000)

hits = 0
for p in points:
    t = (p.payload or {}).get("text") or ""
    if "män" in t.lower() and "buss" in t.lower():
        hits += 1
        print("page", (p.payload or {}).get("page_no"))
        print(t[:1200])
        print("====")
print("hits", hits)


# Run code with: docker compose exec app python data/test/test7.py
