import streamlit as st
from dotenv import load_dotenv

load_dotenv()


# ==============================================
# Imports
# ==============================================

from src.loader import main_loader
from src.splitter import split_documents
from src.embedding import get_embedding
from src.vectorstore import create_vectorstore
from src.pipeline import RAGPipeline


# ==============================================
# Page Configuration
# ==============================================

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🤖",
    layout="wide"
)


# ==============================================
# Title
# ==============================================

st.title("🤖 RAG Assistant")



st.caption(
    "Retrieval-Augmented Generation with "
    "Hybrid Retrieval, Reranking and Citations"
)


# ==============================================
# Initialize RAG Pipeline
# ==============================================

@st.cache_resource
def create_rag_pipeline():

    # ------------------------------------------
    # Load Documents
    # ------------------------------------------

    documents = main_loader(
        "data"
    )

    # ------------------------------------------
    # Split Documents
    # ------------------------------------------

    chunks = split_documents(
        documents
    )

    # ------------------------------------------
    # Embedding
    # ------------------------------------------

    embedding = get_embedding()

    # ------------------------------------------
    # Vector Store
    # ------------------------------------------

    vectorstore = create_vectorstore(
        chunks,
        embedding
    )

    # ------------------------------------------
    # RAG Pipeline
    # ------------------------------------------

    rag = RAGPipeline(
        vectorstore,
        chunks
    )

    return rag


# ==============================================
# Load Pipeline
# ==============================================

with st.spinner(
    "Loading RAG system..."
):

    rag = create_rag_pipeline()


# ==============================================
# Sidebar
# ==============================================

with st.sidebar:

    st.header("⚙️ Settings")

    st.write(
        "RAG Components:"
    )

    st.success(
        "✅ Hybrid Retrieval"
    )

    st.success(
        "✅ BM25"
    )

    st.success(
        "✅ Reranker"
    )

    st.success(
        "✅ Query Rewriting"
    )

    st.success(
        "✅ Conversation Memory"
    )

    st.success(
        "✅ Citations"
    )

    st.success(
        "✅ Streaming"
    )

    st.divider()

    if st.button(
        "🗑️ Clear Conversation"
    ):

        rag.clear_memory()

        st.session_state.messages = []

        st.rerun()


# ==============================================
# Chat History
# ==============================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ==============================================
# Display Previous Messages
# ==============================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        # --------------------------------------
        # Sources
        # --------------------------------------

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

                        metadata = doc.metadata

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

                        st.write(
                            f"**[{i}]** "
                            f"{source} — Page {page}"
                        )


# ==============================================
# User Input
# ==============================================

question = st.chat_input(
    "Ask a question about your documents..."
)


# ==============================================
# Process Question
# ==============================================

if question:

    # ------------------------------------------
    # Display User Message
    # ------------------------------------------

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


    # ------------------------------------------
    # Assistant
    # ------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        answer_placeholder = st.empty()

        source_placeholder = st.empty()

        # --------------------------------------
        # Run RAG
        # --------------------------------------

        with st.spinner(
            "Searching documents..."
        ):

            result = rag.answer(
                question
            )

        answer = result[
            "answer"
        ]

        sources = result[
            "sources"
        ]


        # --------------------------------------
        # Display Answer
        # --------------------------------------

        answer_placeholder.markdown(
            answer
        )


        # --------------------------------------
        # Display Sources
        # --------------------------------------

        if sources:

            with st.expander(
                "📚 Sources"
            ):

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

                    if isinstance(
                        page,
                        int
                    ):

                        page += 1

                    st.write(
                        f"**[{i}]** "
                        f"{source} — Page {page}"
                    )


    # ------------------------------------------
    # Save Assistant Message
    # ------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources
        }
    )