from dotenv import load_dotenv

load_dotenv()


from src.pipeline import RAGPipeline
from src.loader import main_loader
from src.splitter import split_documents
from src.embedding import get_embedding
from src.vectorstore import create_vectorstore


def main():

    print("=" * 50)
    print("شروع پروژه RAG")
    print("=" * 50)

    # ==========================================
    # 1. Load Documents
    # ==========================================

    print("\n[1] Loading PDF...")

    documents = main_loader(
        "data"
    )

    print(
        f"تعداد صفحات: {len(documents)}"
    )

    # ==========================================
    # 2. Split Documents
    # ==========================================

    print("\n[2] Splitting Documents...")

    chunks = split_documents(
        documents
    )

    print(
        f"تعداد Chunkها: {len(chunks)}"
    )

    # ==========================================
    # 3. Embedding
    # ==========================================

    print("\n[3] Creating Embedding Model...")

    embedding = get_embedding()

    # ==========================================
    # 4. Vector Store
    # ==========================================

    print("\n[4] Creating Vector Store...")

    vectorstore = create_vectorstore(
        chunks,
        embedding
    )

    print(
        "✅ Vector Store ساخته شد."
    )

    # ==========================================
    # 5. RAG Pipeline
    # ==========================================

    print("\n[5] Creating RAG Pipeline...")

    rag = RAGPipeline(
        vectorstore,
        chunks
    )

    print(
        "\nبرای خروج exit یا quit وارد کنید."
    )

    print(
        "برای پاک کردن Memory عبارت clear را وارد کنید.\n"
    )

    # ==========================================
    # 6. Chat Loop
    # ==========================================

    while True:

        question = input(
            "سؤال خود را وارد کنید: "
        ).strip()

        # ======================================
        # Exit
        # ======================================

        if question.lower() in [
            "exit",
            "quit"
        ]:

            print(
                "خروج از برنامه..."
            )

            break

        # ======================================
        # Clear Memory
        # ======================================

        if question.lower() == "clear":

            rag.clear_memory()

            continue

        # ======================================
        # Empty Question
        # ======================================

        if not question:

            continue

        # ======================================
        # Run RAG Pipeline
        # ======================================

        print(
            "\nدر حال پردازش...\n"
        )

        # مهم:
        # main.py از answer() استفاده می‌کند
        # stream_answer() برای Streamlit است.

        result = rag.answer(
            question
        )

        # ======================================
        # Answer
        # ======================================

        answer = result["answer"]

        sources = result["sources"]

        print(
            "\n" + "=" * 60
        )

        print(
            "پاسخ مدل"
        )

        print(
            "=" * 60
        )

        print(
            answer
        )

        # ======================================
        # Sources
        # ======================================

        print(
            "\n" + "=" * 60
        )

        print(
            "Sources"
        )

        print(
            "=" * 60
        )

        if not sources:

            print(
                "No sources found."
            )

        else:

            for i, doc in enumerate(
                sources,
                start=1
            ):

                metadata = doc.metadata

                source = metadata.get(
                    "source",
                    "Unknown"
                )

                page = metadata.get(
                    "page",
                    "Unknown"
                )

                # PyPDF page index معمولاً از 0 شروع می‌شود
                if isinstance(page, int):

                    page = page + 1

                print(
                    f"\n[{i}]"
                )

                print(
                    f"File: {source}"
                )

                print(
                    f"Page: {page}"
                )

        print()


# ==============================================
# Run Application
# ==============================================

if __name__ == "__main__":



    
    main()