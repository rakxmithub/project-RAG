from langchain_classic.retrievers import EnsembleRetriever
from src.bm25 import create_bm25_retriever


def create_hybrid_retriever(vectorstore, chunks):

    vector_retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 5
        }
    )

    bm25_retriever = create_bm25_retriever(chunks)

    hybrid_retriever = EnsembleRetriever(
        retrievers=[
            vector_retriever,
            bm25_retriever
        ],
        weights=[
            0.5,
            0.5
        ]
    )

    return hybrid_retriever