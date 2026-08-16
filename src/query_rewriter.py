from src.llm import get_llm


class QueryRewriter:

    def __init__(self):

        self.llm = get_llm()


    def rewrite(self, question, history=None):

        # ==========================================
        # Build conversation history
        # ==========================================

        history_text = ""

        if history:

            history_parts = []

            for message in history:

                role = message.get("role", "")
                content = message.get("content", "")

                if role == "user":

                    history_parts.append(
                        f"User: {content}"
                    )

                elif role == "assistant":

                    history_parts.append(
                        f"Assistant: {content}"
                    )

            history_text = "\n".join(
                history_parts
            )


        # ==========================================
        # Query Rewriting Prompt
        # ==========================================

        prompt = f"""
You are a query rewriting system for a
Retrieval-Augmented Generation (RAG) system.

Your job is to rewrite the user's current question
into a clear, self-contained search query.

Conversation history:
--------------------
{history_text}
--------------------

Current user question:
--------------------
{question}
--------------------

Rules:

1. Preserve the original meaning.
2. Use conversation history when necessary.
3. Resolve references such as:
   "it", "its", "they", "them", "this", "that".
4. Make the query self-contained.
5. Keep important technical terms.
6. Do not answer the question.
7. Do not add unsupported information.
8. If the question is already clear, keep it mostly unchanged.
9. Return ONLY the rewritten search query.
10. Do not add explanations.

Rewritten search query:
"""


        # ==========================================
        # LLM
        # ==========================================

        response = self.llm.invoke(
            prompt
        )


        # ==========================================
        # Clean result
        # ==========================================

        rewritten_query = response.content.strip()

        return rewritten_query