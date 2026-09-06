from src.llm import get_llm


class QueryRewriter:

    def __init__(self):

        print("Initializing Query Rewriter...")

        self.llm = get_llm()


    # =========================================================
    # REWRITE
    # =========================================================

    def rewrite(self, question):

        prompt = f"""
You are a query rewriting assistant for a Retrieval-Augmented
Generation system.

Rewrite the user's question into a clear, standalone search query.

Rules:

1. Preserve the original meaning.
2. Expand abbreviations when useful.
3. If the question refers to something implicitly, make it explicit.
4. Do not answer the question.
5. Do not add information that is not implied by the question.
6. Return ONLY the rewritten query.
7. Do not use quotes.
8. Do not explain your changes.

Original question:
{question}

Rewritten query:
"""

        response = self.llm.invoke(prompt)

        rewritten = response.content.strip()

        return rewritten