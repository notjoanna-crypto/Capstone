from qdrant_client import QdrantClient

COLLECTION = "resvaneundersokning_ComponentChange"
c = QdrantClient(host="qdrant", port=6333)

points, next_page = c.scroll(
    collection_name=COLLECTION,
    with_payload=True,
    with_vectors=False,
    limit=5,
)

print("n_points", len(points), "next_page", next_page)

for i, p in enumerate(points):
    payload = p.payload or {}
    print("----", i)
    print("payload_keys", list(payload.keys()))
    md = payload.get("metadata", {}) or {}
    print("metadata_keys", list(md.keys()) if isinstance(md, dict) else type(md))
    for k in ["text", "content", "page_content"]:
        if k in payload and payload[k]:
            print("found", k, "len", len(str(payload[k])))
            print(str(payload[k])[:500])



# Run code with: docker compose exec app python data/test/test5.py
