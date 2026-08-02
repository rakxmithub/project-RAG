from langchain_mistralai import ChatMistralAI

# Get LLM Function
def get_llm():
    llm = ChatMistralAI(
        model="mistral-small-latest"
    )

    return llm