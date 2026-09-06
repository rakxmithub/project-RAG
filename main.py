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


    # ========================================================
    # 1. LOAD DOCUMENTS
    # ========================================================

    print("\n[1] Loading PDF...")

    documents = main_loader(
        "data"
    )

    print(
        f"تعداد صفحات: {len(documents)}"
    )


    # ========================================================
    # 2. SPLIT
    # ========================================================

    print(
        "\n[2] Splitting Documents..."
    )

    chunks = split_documents(
        documents
    )

    print(
        f"تعداد Chunkها: {len(chunks)}"
    )


    # ========================================================
    # 3. EMBEDDING
    # ========================================================

    print(
        "\n[3] Creating Embedding Model..."
    )

    embedding = get_embedding()


    # ========================================================
    # 4. VECTOR STORE
    # ========================================================

    print(
        "\n[4] Creating Vector Store..."
    )

    vectorstore = create_vectorstore(
        chunks,
        embedding
    )

    print(
        "✅ Vector Store ساخته شد."
    )


    # ========================================================
    # 5. PIPELINE
    # ========================================================

    print(
        "\n[5] Creating RAG Pipeline..."
    )

    rag = RAGPipeline(
        vectorstore,
        chunks
    )


    # ========================================================
    # CHAT
    # ========================================================

    print(
        "\nبرای خروج exit یا quit وارد کنید."
    )

    print(
        "برای پاک کردن Memory عبارت clear را وارد کنید.\n"
    )


    while True:

        question = input(
            "سؤال خود را وارد کنید: "
        ).strip()


        # ====================================================
        # EXIT
        # ====================================================

        if question.lower() in [
            "exit",
            "quit"
        ]:

            print(
                "خروج از برنامه..."
            )

            break


        # ====================================================
        # CLEAR MEMORY
        # ====================================================

        if question.lower() == "clear":

            rag.clear_memory()

            continue


        # ====================================================
        # EMPTY
        # ====================================================

        if not question:

            continue


        # ====================================================
        # PROCESS
        # ====================================================

        print(
            "\nدر حال پردازش...\n"
        )


        try:

            result = rag.answer(
                question
            )


        except Exception as e:

            print(
                "\n❌ Error:"
            )

            print(
                str(e)
            )

            continue


        # ====================================================
        # ANSWER
        # ====================================================

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
            result["answer"]
        )


        # ====================================================
        # SOURCES
        # ====================================================

        print(
            "\n" + "=" * 60
        )

        print(
            "Sources"
        )

        print(
            "=" * 60
        )


        sources = result.get(
            "sources",
            []
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

                metadata = (
                    doc.metadata
                )


                source = metadata.get(
                    "source",
                    "Unknown"
                )


                page = metadata.get(
                    "page",
                    "Unknown"
                )


                # PyPDF usually starts at 0
                if isinstance(
                    page,
                    int
                ):

                    page += 1


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


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()