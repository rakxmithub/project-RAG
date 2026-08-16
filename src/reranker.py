from sentence_transformers import CrossEncoder


def get_reranker():
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return model


def rerank_documents(question, docs, reranker, top_k=3):
    pairs = []

    for doc in docs:
        pairs.append([question, doc.page_content])

    scores = reranker.predict(pairs)

    ranked_docs = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [doc for doc, score in ranked_docs[:top_k]]