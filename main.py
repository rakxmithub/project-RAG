from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.loader import main_loader
from src.splitter import split_documents
from src.embedding import get_embedding
from src.vectorstore import create_vectorstore
from src.retriever import get_retriever
from src.llm import get_llm


def main():
    print("=" * 50)
    print("شروع پروژه RAG")
    print("=" * 50)

    # 1. Load PDF
    print("\n[1] Loading PDF...")
    documents = main_loader("data")
    print(f"تعداد صفحات: {len(documents)}")

    # 2. Split Documents
    print("\n[2] Splitting Documents...")
    chunks = split_documents(documents)
    print(f"تعداد Chunkها: {len(chunks)}")

    # 3. Embedding
    print("\n[3] Creating Embedding Model...")
    embedding = get_embedding()

    # 4. Vector Store
    print("\n[4] Creating Vector Store...")
    vectorstore = create_vectorstore(chunks, embedding)
    print("✅ Vector Store ساخته شد.")

    # 5. Retriever
    print("\n[5] Creating Retriever...")
    retriever = get_retriever(vectorstore)

    # 6. Ask Question
    question = input("\nسؤال خود را وارد کنید: ")
    print(f"\nQuestion: {question}")

    docs = retriever.invoke(question)

    if not docs:
        print("هیچ سندی پیدا نشد.")
        return

    print(f"\n{len(docs)} Chunk پیدا شد.\n")

    for i, doc in enumerate(docs, start=1):
        print("=" * 60)
        print(f"Chunk {i}")
        print("=" * 60)
        print(doc.page_content[:400])
        print()

    # 7. LLM
    print("\n[6] Loading LLM...")
    llm = get_llm()

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
You are a helpful AI assistant.

Answer ONLY using the provided context.
If the answer is not in the context, say:
"I don't know based on the provided context."

Context:
{context}

Question:
{question}

Answer:
"""

    print("\nGenerating answer...\n")

    response = llm.invoke(prompt)

    print("=" * 60)
    print("پاسخ مدل")
    print("=" * 60)
    print(response.content)


if __name__ == "__main__":
    main()


