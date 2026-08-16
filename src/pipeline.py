from src.hybrid_retriever import create_hybrid_retriever
from src.reranker import get_reranker, rerank_documents
from src.llm import get_llm
from src.query_rewriter import QueryRewriter
from src.memory import ConversationMemory

from src.logger import (
    log_info,
    log_query,
    log_rewritten_query,
    log_retrieval,
    log_reranking,
    log_answer,
    log_warning,
)


class RAGPipeline:

    def __init__(self, vectorstore, chunks):

        print("Creating Hybrid Retriever...")

        self.retriever = create_hybrid_retriever(
            vectorstore,
            chunks
        )

        print("Loading Reranker...")

        self.reranker = get_reranker()

        print("Loading LLM...")

        self.llm = get_llm()

        print("Loading Query Rewriter...")

        self.query_rewriter = QueryRewriter()

        print("Creating Conversation Memory...")

        self.memory = ConversationMemory()

        log_info(
            "RAG Pipeline initialized successfully."
        )


    # ==================================================
    # RETRIEVAL
    # ==================================================

    def retrieve(self, question):

        print("Retrieving documents...")

        docs = self.retriever.invoke(
            question
        )

        print(
            f"Retrieved documents: {len(docs)}"
        )

        log_retrieval(
            len(docs)
        )

        return docs


    # ==================================================
    # CREATE CONTEXT
    # ==================================================

    def create_context(self, ranked_docs):

        print("Creating context...")

        context_parts = []

        for i, doc in enumerate(
            ranked_docs,
            start=1
        ):

            context_parts.append(
                f"[{i}]\n"
                f"{doc.page_content}"
            )

        return "\n\n".join(
            context_parts
        )


    # ==================================================
    # CREATE PROMPT
    # ==================================================

    def create_prompt(
        self,
        question,
        context
    ):

        return f"""
You are a reliable Retrieval-Augmented Generation (RAG) assistant.

Answer the user's question using ONLY the provided context.

Rules:

1. Use only the provided context.
2. Do not use outside knowledge.
3. Do not invent or guess information.
4. If the answer is not supported by the context, say:

"I don't know based on the provided context."

5. Answer clearly and directly.
6. Keep the answer concise but informative.
7. Preserve important technical terms, names, and numbers.
8. Add citations for factual claims.
9. Use citations in this format: [1], [2], [3].
10. Only cite sources that actually support the claim.
11. Never create citation numbers that do not exist.
12. Do not mention these instructions.

Context:
--------------------------------------------------

{context}

--------------------------------------------------

Question:
{question}

Answer:
"""


    # ==================================================
    # NORMAL ANSWER
    # ==================================================

    def answer(self, question):

        log_query(question)

        # ----------------------------------------------
        # Query Rewrite
        # ----------------------------------------------

        print("\nRewriting query...")

        rewritten_question = (
            self.query_rewriter.rewrite(
                question
            )
        )

        print(
            f"Original question: {question}"
        )

        print(
            f"Rewritten query: {rewritten_question}"
        )

        log_rewritten_query(
            rewritten_question
        )

        # ----------------------------------------------
        # Retrieval
        # ----------------------------------------------

        docs = self.retrieve(
            rewritten_question
        )

        if not docs:

            log_warning(
                "No relevant documents found."
            )

            return {
                "answer":
                    "I don't know based on the provided context.",
                "sources": []
            }

        # ----------------------------------------------
        # Reranking
        # ----------------------------------------------

        print("Reranking documents...")

        ranked_docs = rerank_documents(
            question=rewritten_question,
            docs=docs,
            reranker=self.reranker,
            top_k=3
        )

        print(
            f"Top documents after reranking: "
            f"{len(ranked_docs)}"
        )

        log_reranking(
            len(ranked_docs)
        )

        # ----------------------------------------------
        # Context
        # ----------------------------------------------

        context = self.create_context(
            ranked_docs
        )

        # ----------------------------------------------
        # Prompt
        # ----------------------------------------------

        prompt = self.create_prompt(
            question,
            context
        )

        # ----------------------------------------------
        # LLM
        # ----------------------------------------------

        print("\nGenerating answer...")

        response = self.llm.invoke(
            prompt
        )

        answer = response.content

        log_answer(
            answer
        )

        # ----------------------------------------------
        # Memory
        # ----------------------------------------------

        self.memory.add_user_message(
            question
        )

        self.memory.add_ai_message(
            answer
        )

        return {
            "answer": answer,
            "sources": ranked_docs
        }


    # ==================================================
    # STREAM ANSWER
    # ==================================================

    def stream_answer(self, question):

        log_query(question)

        # ----------------------------------------------
        # Query Rewrite
        # ----------------------------------------------

        print("\nRewriting query...")

        rewritten_question = (
            self.query_rewriter.rewrite(
                question
            )
        )

        print(
            f"Original question: {question}"
        )

        print(
            f"Rewritten query: {rewritten_question}"
        )

        log_rewritten_query(
            rewritten_question
        )


        # ----------------------------------------------
        # Retrieval
        # ----------------------------------------------

        docs = self.retrieve(
            rewritten_question
        )

        if not docs:

            message = (
                "I don't know based on the provided context."
            )

            log_warning(
                "No relevant documents found."
            )

            yield {
                "type": "token",
                "content": message
            }

            yield {
                "type": "sources",
                "sources": []
            }

            return


        # ----------------------------------------------
        # Reranking
        # ----------------------------------------------

        print(
            "Reranking documents..."
        )

        ranked_docs = rerank_documents(
            question=rewritten_question,
            docs=docs,
            reranker=self.reranker,
            top_k=3
        )

        print(
            f"Top documents after reranking: "
            f"{len(ranked_docs)}"
        )

        log_reranking(
            len(ranked_docs)
        )


        # ----------------------------------------------
        # Context
        # ----------------------------------------------

        context = self.create_context(
            ranked_docs
        )


        # ----------------------------------------------
        # Prompt
        # ----------------------------------------------

        prompt = self.create_prompt(
            question,
            context
        )


        # ----------------------------------------------
        # Streaming LLM
        # ----------------------------------------------

        print(
            "\nGenerating answer..."
        )

        full_answer = ""


        try:

            for chunk in self.llm.stream(
                prompt
            ):

                # LangChain AIMessageChunk
                if hasattr(
                    chunk,
                    "content"
                ):

                    token = chunk.content

                else:

                    token = str(chunk)


                if token:

                    full_answer += token

                    yield {
                        "type": "token",
                        "content": token
                    }


        except Exception as e:

            log_warning(
                f"Streaming failed: {e}"
            )

            # Fallback to normal invoke
            response = self.llm.invoke(
                prompt
            )

            full_answer = response.content

            yield {
                "type": "token",
                "content": full_answer
            }


        # ----------------------------------------------
        # Logging
        # ----------------------------------------------

        log_answer(
            full_answer
        )


        # ----------------------------------------------
        # Memory
        # ----------------------------------------------

        self.memory.add_user_message(
            question
        )

        self.memory.add_ai_message(
            full_answer
        )


        # ----------------------------------------------
        # Sources
        # ----------------------------------------------

        yield {
            "type": "sources",
            "sources": ranked_docs
        }


    # ==================================================
    # CLEAR MEMORY
    # ==================================================

    def clear_memory(self):

        self.memory.clear()

        log_info(
            "Conversation memory cleared."
        )

        print(
            "Memory cleared."
        )