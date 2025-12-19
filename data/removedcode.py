answer = result["generator"]
retrieved_texts = [chunk.text for chunk in retrieved_chunks]

sources = [chunk.metadata.get("doc_id", "unknown") for chunk in retrieved_chunks]
record = {
    "question": query,
    "answer": answer,
    "retrieved_text": retrieved_texts,
    "sources": sources,
    "configuration": "clean_rag"
}

#print(record)