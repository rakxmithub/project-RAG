from langchain_mistralai import MistralAIEmbeddings

def get_embedding():
    return MistralAIEmbeddings(
        model="mistral-embed"
    )