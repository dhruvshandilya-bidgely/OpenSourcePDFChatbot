import os
import langchain_core.documents.base

def see_docs(docs: list[langchain_core.documents.base.Document]):
    """
    Saves the docs as .txt in DEBUG/DOCS folder

    Args:
        docs: (list[langchain_core.documents.base.Document]) Is a list of langchain documents of our pdf.

    Returns:
        None.
    """

    cwd = os.getcwd()

    folder = cwd + '/DEBUG/DOCS'

    for i in range(len(docs)):
        f = open(f"{folder}/{i}.txt", "w")
        f.write(docs[i].page_content)
        f.close()