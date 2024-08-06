from langchain.text_splitter import RecursiveCharacterTextSplitter
import langchain_core.documents.base

def document_from_text(pdf_text: str, chunk_size: int = 2500, chunk_overlap: int = 200) -> list[langchain_core.documents.base.Document]:
    """
    Converts the pdf text we obtained to list of langchain docs which contains chunks obtained recursively.

    Args:
        pdf_text: (String) The textual content of pdf.
        chunk_size: (int) Token size of each chunk that is to be formed.
        chunk_overlap: (int) How many tokens should overlap between two chunks.
 
    Returns:
        docs : (langchain_core.documents.base.Document) Returns list of Langchain docs.
    """
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=2500, chunk_overlap=200)
    docs = text_splitter.create_documents([pdf_text])
    return docs