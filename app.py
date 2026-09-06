import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.loader import main_loader
from src.splitter import split_documents
from src.embedding import get_embedding
from src.vectorstore import create_vectorstore
from src.pipeline import RAGPipeline


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🤖 RAG Assistant")

st.caption(
    "Ask questions about your documents using Retrieval-Augmented Generation."
)


# ============================================================
# SESSION STATE
# ============================================================

if "rag" not in st.session_state:

    with st.spinner(
        "Loading RAG system..."
    ):

        # ----------------------------------------------------
        # Load documents
        # ----------------------------------------------------

        documents = main_loader(
            "data"
        )

        # ----------------------------------------------------
        # Split
        # ----------------------------------------------------

        chunks = split_documents(
            documents
        )

        # ----------------------------------------------------
        # Embedding
        # ----------------------------------------------------

        embedding = get_embedding()

        # ----------------------------------------------------
        # Vector Store
        # ----------------------------------------------------

        vectorstore = create_vectorstore(
            chunks,
            embedding
        )

        # ----------------------------------------------------
        # Pipeline
        # ----------------------------------------------------

        st.session_state.rag = RAGPipeline(
            vectorstore,
            chunks
        )


# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Controls")

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.session_state.rag.clear_memory()

        st.rerun()


    st.divider()

    st.subheader("📊 System")

    st.write(
        "RAG Pipeline: ✅"
    )

    st.write(
        "Hybrid Retrieval: ✅"
    )

    st.write(
        "Reranker: ✅"
    )

    st.write(
        "Query Rewriting: ✅"
    )

    st.write(
        "Citation Tracking: ✅"
    )

    st.write(
        "Conversation Memory: ✅"
    )

    st.write(
        "Logging: ✅"
    )


# ============================================================
# DISPLAY PREVIOUS MESSAGES
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        # ----------------------------------------------------
        # Sources
        # ----------------------------------------------------

        if (
            message["role"] == "assistant"
            and "sources" in message
        ):

            sources = message["sources"]

            if sources:

                with st.expander(
                    "📚 Sources"
                ):

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

                        if isinstance(
                            page,
                            int
                        ):

                            page += 1

                        st.markdown(
                            f"""
                            **[{i}] {source}**  
                            Page: {page}
                            """
                        )


# ============================================================
# USER INPUT
# ============================================================

question = st.chat_input(
    "سؤال خود را درباره اسناد وارد کنید..."
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    # --------------------------------------------------------
    # Display user message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )


    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )


    # --------------------------------------------------------
    # Assistant
    # --------------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        answer_placeholder = st.empty()

        full_answer = ""

        sources = []


        try:

            # =================================================
            # RUN RAG
            # =================================================

            result = st.session_state.rag.answer(
                question
            )


            full_answer = result[
                "answer"
            ]

            sources = result.get(
                "sources",
                []
            )


            # =================================================
            # SHOW ANSWER
            # =================================================

            answer_placeholder.markdown(
                full_answer
            )


            # =================================================
            # SHOW SOURCES
            # =================================================

            if sources:

                with st.expander(
                    "📚 Sources"
                ):

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

                        if isinstance(
                            page,
                            int
                        ):

                            page += 1


                        st.markdown(
                            f"""
                            **[{i}] {source}**  
                            Page: {page}
                            """
                        )


        except Exception as e:

            full_answer = (
                f"❌ Error: {str(e)}"
            )

            answer_placeholder.error(
                full_answer
            )


    # --------------------------------------------------------
    # Save conversation
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_answer,
            "sources": sources
        }
    )