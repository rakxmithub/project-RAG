from langchain_mistralai import ChatMistralAI


def get_llm():
    llm = ChatMistralAI(
        model="mistral-small-latest"
    )

    return llm