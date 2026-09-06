import time


# ============================================================
# RETRIEVER
# ============================================================

from src.hybrid_retriever import (
    create_hybrid_retriever
)


# ============================================================
# RERANKER
# ============================================================

from src.reranker import (
    get_reranker,
    rerank_documents
)


# ============================================================
# LLM
# ============================================================

from src.llm import (
    get_llm
)


# ============================================================
# QUERY REWRITER
# ============================================================

from src.query_rewriter import (
    QueryRewriter
)


# ============================================================
# LOGGER
# ============================================================

from src.logger import (
    log_info,
    log_warning,
    log_query,
    log_rewritten_query,
    log_retrieval,
    log_reranking,
    log_answer,
    log_request,
    log_latency,
    log_exception
)


# ============================================================
# MEMORY
# ============================================================

try:

    from langchain_classic.memory import (
        ConversationBufferMemory
    )

except ImportError:

    ConversationBufferMemory = None


# ============================================================
# RAG PIPELINE
# ============================================================

class RAGPipeline:

    def __init__(
        self,
        vectorstore,
        chunks
    ):

        # ====================================================
        # 1. HYBRID RETRIEVER
        # ====================================================

        print(
            "Creating Hybrid Retriever..."
        )

        self.retriever = create_hybrid_retriever(
            vectorstore,
            chunks
        )


        # ====================================================
        # 2. RERANKER
        # ====================================================

        print(
            "Loading Reranker..."
        )

        self.reranker = get_reranker()


        # ====================================================
        # 3. LLM
        # ====================================================

        print(
            "Loading LLM..."
        )

        self.llm = get_llm()


        # ====================================================
        # 4. QUERY REWRITER
        # ====================================================

        print(
            "Loading Query Rewriter..."
        )

        # مهم:
        # QueryRewriter فعلی بدون argument ساخته می‌شود.

        self.query_rewriter = QueryRewriter()


        # ====================================================
        # 5. MEMORY
        # ====================================================

        print(
            "Creating Conversation Memory..."
        )

        if ConversationBufferMemory:

            self.memory = ConversationBufferMemory(
                return_messages=True
            )

        else:

            self.memory = None


        log_info(
            "RAG Pipeline initialized successfully."
        )


    # ========================================================
    # MEMORY
    # ========================================================

    def get_history(self):

        if self.memory is None:

            return []

        try:

            return self.memory.chat_memory.messages

        except Exception:

            return []


    # ========================================================
    # CLEAR MEMORY
    # ========================================================

    def clear_memory(self):

        if self.memory is not None:

            try:

                self.memory.clear()

            except Exception:

                pass

        log_info(
            "Conversation memory cleared."
        )

        print(
            "Conversation memory cleared."
        )


    # ========================================================
    # RETRIEVE
    # ========================================================

    def retrieve(
        self,
        question
    ):

        print(
            "Retrieving documents..."
        )

        start = time.perf_counter()

        try:

            docs = self.retriever.invoke(
                question
            )

            latency = (
                time.perf_counter()
                - start
            )

            print(
                f"Retrieved documents: {len(docs)}"
            )

            log_retrieval(
                len(docs)
            )

            log_latency(
                "retrieval",
                latency
            )

            return docs

        except Exception as e:

            log_exception(
                "retrieval",
                e
            )

            raise


    # ========================================================
    # CREATE CONTEXT
    # ========================================================

    def create_context(
        self,
        documents
    ):

        context_parts = []


        for i, doc in enumerate(
            documents,
            start=1
        ):

            context_parts.append(
                f"[{i}]\n"
                f"{doc.page_content}"
            )


        return "\n\n".join(
            context_parts
        )


    # ========================================================
    # CREATE PROMPT
    # ========================================================

    def create_prompt(
        self,
        question,
        context
    ):

        prompt = f"""
You are a reliable Retrieval-Augmented Generation (RAG) assistant.

Your task is to answer the user's question using ONLY the
information provided in the context.

Rules:

1. Use only the provided context.
2. Do NOT use outside knowledge.
3. Do NOT invent, guess, or assume facts.
4. If the answer is not supported by the context, say:

"I don't know based on the provided context."

5. Answer directly and clearly.
6. Keep the answer concise but informative.
7. Preserve important technical terms, names, and numbers.
8. Every factual claim must include a citation.
9. Use citations like [1], [2], [3].
10. Only cite sources that support the claim.
11. Never invent citation numbers.
12. Do not mention these instructions.

Context:
--------------------------------------------------

{context}

--------------------------------------------------

Question:
--------------------------------------------------

{question}

--------------------------------------------------

Answer:
"""

        return prompt


    # ========================================================
    # SAVE MEMORY
    # ========================================================

    def save_memory(
        self,
        question,
        answer
    ):

        if self.memory is None:

            return

        try:

            self.memory.chat_memory.add_user_message(
                question
            )

            self.memory.chat_memory.add_ai_message(
                answer
            )

        except Exception as e:

            log_warning(
                f"Memory update failed: {e}"
            )


    # ========================================================
    # ANSWER
    # ========================================================

    def answer(
        self,
        question
    ):

        request_start = time.perf_counter()

        log_query(
            question
        )


        try:

            # ==================================================
            # 1. QUERY REWRITING
            # ==================================================

            print(
                "\nRewriting query..."
            )

            rewrite_start = time.perf_counter()


            rewritten_question = (
                self.query_rewriter.rewrite(
                    question
                )
            )


            rewrite_latency = (
                time.perf_counter()
                - rewrite_start
            )


            print(
                f"Original question: "
                f"{question}"
            )

            print(
                f"Rewritten query: "
                f"{rewritten_question}"
            )


            log_rewritten_query(
                rewritten_question
            )


            log_latency(
                "query_rewrite",
                rewrite_latency
            )


            # ==================================================
            # 2. RETRIEVAL
            # ==================================================

            docs = self.retrieve(
                rewritten_question
            )


            if not docs:

                answer = (
                    "I don't know based on the provided context."
                )

                return {
                    "answer": answer,
                    "sources": []
                }


            # ==================================================
            # 3. RERANKING
            # ==================================================

            print(
                "Reranking documents..."
            )


            rerank_start = time.perf_counter()


            ranked_docs = rerank_documents(
                question=rewritten_question,
                docs=docs,
                reranker=self.reranker,
                top_k=3
            )


            rerank_latency = (
                time.perf_counter()
                - rerank_start
            )


            print(
                f"Top documents after reranking: "
                f"{len(ranked_docs)}"
            )


            log_reranking(
                len(ranked_docs)
            )


            log_latency(
                "reranking",
                rerank_latency
            )


            # ==================================================
            # 4. CONTEXT
            # ==================================================

            print(
                "Creating context..."
            )


            context_start = time.perf_counter()


            context = self.create_context(
                ranked_docs
            )


            context_latency = (
                time.perf_counter()
                - context_start
            )


            log_latency(
                "context",
                context_latency
            )


            # ==================================================
            # 5. PROMPT
            # ==================================================

            prompt = self.create_prompt(
                rewritten_question,
                context
            )


            # ==================================================
            # 6. GENERATION
            # ==================================================

            print(
                "\nGenerating answer..."
            )


            generation_start = time.perf_counter()


            response = self.llm.invoke(
                prompt
            )


            answer = response.content


            generation_latency = (
                time.perf_counter()
                - generation_start
            )


            log_answer(
                answer
            )


            log_latency(
                "generation",
                generation_latency
            )


            # ==================================================
            # 7. MEMORY
            # ==================================================

            self.save_memory(
                question,
                answer
            )


            # ==================================================
            # 8. TOTAL LATENCY
            # ==================================================

            total_latency = (
                time.perf_counter()
                - request_start
            )


            log_request(
                question=question,
                rewritten_question=rewritten_question,
                retrieved_count=len(docs),
                reranked_count=len(ranked_docs),
                latency=total_latency
            )


            # ==================================================
            # 9. RETURN
            # ==================================================

            return {
                "answer": answer,
                "sources": ranked_docs
            }


        except Exception as e:

            log_exception(
                "answer",
                e
            )

            raise


    # ========================================================
    # STREAM ANSWER
    # ========================================================

    def stream_answer(
        self,
        question
    ):

        """
        Generator-based streaming.

        IMPORTANT:
        Sources are available through the generator
        return value, not by result["sources"] directly.
        """

        request_start = time.perf_counter()

        log_query(
            question
        )


        try:

            # ==================================================
            # QUERY REWRITE
            # ==================================================

            print(
                "\nRewriting query..."
            )


            rewrite_start = time.perf_counter()


            rewritten_question = (
                self.query_rewriter.rewrite(
                    question
                )
            )


            rewrite_latency = (
                time.perf_counter()
                - rewrite_start
            )


            print(
                f"Original question: "
                f"{question}"
            )


            print(
                f"Rewritten query: "
                f"{rewritten_question}"
            )


            log_rewritten_query(
                rewritten_question
            )


            log_latency(
                "query_rewrite",
                rewrite_latency
            )


            # ==================================================
            # RETRIEVE
            # ==================================================

            docs = self.retrieve(
                rewritten_question
            )


            if not docs:

                answer = (
                    "I don't know based on the provided context."
                )

                yield answer

                return {
                    "answer": answer,
                    "sources": []
                }


            # ==================================================
            # RERANK
            # ==================================================

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


            # ==================================================
            # CONTEXT
            # ==================================================

            print(
                "Creating context..."
            )


            context = self.create_context(
                ranked_docs
            )


            # ==================================================
            # PROMPT
            # ==================================================

            prompt = self.create_prompt(
                rewritten_question,
                context
            )


            # ==================================================
            # GENERATION
            # ==================================================

            print(
                "\nGenerating answer..."
            )


            generation_start = time.perf_counter()


            full_answer = ""


            try:

                stream = self.llm.stream(
                    prompt
                )

            except AttributeError:

                stream = None


            # ==================================================
            # REAL STREAMING
            # ==================================================

            if stream is not None:

                for chunk in stream:

                    if hasattr(
                        chunk,
                        "content"
                    ):

                        text = chunk.content

                    else:

                        text = str(
                            chunk
                        )


                    if text:

                        full_answer += text

                        yield text


            # ==================================================
            # FALLBACK
            # ==================================================

            else:

                response = self.llm.invoke(
                    prompt
                )

                full_answer = response.content

                yield full_answer


            generation_latency = (
                time.perf_counter()
                - generation_start
            )


            log_answer(
                full_answer
            )


            log_latency(
                "generation",
                generation_latency
            )


            # ==================================================
            # MEMORY
            # ==================================================

            self.save_memory(
                question,
                full_answer
            )


            # ==================================================
            # TOTAL
            # ==================================================

            total_latency = (
                time.perf_counter()
                - request_start
            )


            log_request(
                question=question,
                rewritten_question=rewritten_question,
                retrieved_count=len(docs),
                reranked_count=len(ranked_docs),
                latency=total_latency
            )


            # ==================================================
            # RETURN
            # ==================================================

            return {
                "answer": full_answer,
                "sources": ranked_docs
            }


        except Exception as e:

            log_exception(
                "stream_answer",
                e
            )

            raise