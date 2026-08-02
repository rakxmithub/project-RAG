from langchain_community.document_loaders import PyPDFLoader, TextLoader,   UnstructuredMarkdownLoader
import os


def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

def load_text(file_path):
    loader = TextLoader(file_path,encoding="utf-8")
    documents = loader.load()
    return documents



def load_md(file_path):
    loader = UnstructuredMarkdownLoader(file_path)
    documents = loader.load()
    return documents


def main_loader(folder_path):

    documents = []

    for file in os.listdir(folder_path):

        file_path = os.path.join(folder_path, file)

        if file.endswith(".pdf"):
            documents.extend(load_pdf(file_path))

        elif file.endswith(".txt"):
            documents.extend(load_text(file_path))

        elif file.endswith(".md"):
            documents.extend(load_md(file_path))

    return documents