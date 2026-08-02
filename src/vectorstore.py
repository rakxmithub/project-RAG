from langchain_chroma import Chroma


def create_vectorstore(chunks, embedding):
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory="chroma_db",
    )

    return vectorstore