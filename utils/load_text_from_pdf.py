from langchain_community.document_loaders import PDFPlumberLoader

def text_from_pdf_loader(pdf_location: str) -> str:
    """
    Function to load text from pdf to a string which is returned.

    Args:
        pdf_location: (String) The location of the pdf.

    Returns:
        pdf_text: (String) Returns text of the pdf in string format.
    """
    loader = PDFPlumberLoader(pdf_location)
    pagewise_text = loader.load()

    pdf_text = ""

    for i in pagewise_text:
        pdf_text += i.page_content

    return pdf_text