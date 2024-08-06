from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
import langchain_community.vectorstores.faiss

from utils.load_text_from_pdf import text_from_pdf_loader
from utils.docs_from_text_string import document_from_text

def create_vectorstore(pdf_loc_name: str) -> langchain_community.vectorstores.faiss.FAISS:
    """
    Creates a vector based on embeddings from hugging face to carry RAG over.

    Args:
        pdf_loc_name : (str) The location of pdf in string format.

    Returns:
        vectorstore : (langchain_community.vectorstores.faiss.FAISS) returns the FAISS vectorstore.
    """

    txt_str = text_from_pdf_loader(pdf_loc_name)

    documents = document_from_text(pdf_text=txt_str, chunk_size=1000, chunk_overlap=100)

    # Instantiate the embedding model
    embedder = HuggingFaceEmbeddings()

    # Create the vector store 
    vectorstore = FAISS.from_documents(documents, embedder)

    return vectorstore

def rag_bot(vectorstore: langchain_community.vectorstores.faiss.FAISS, question: str):
    """
    Creates a vector based on embeddings from hugging face to carry RAG over.

    Args:
        vectorstore : (langchain_community.vectorstores.faiss.FAISS) The vectorstore we want to do RAG over.
        question : (str) The question we want an answer to.
        
    Returns:
        vector : 
    """

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    prompt = """
    1. Use the following pieces of context to answer the question at the end.
    2. If you don't know the answer or cannot answer, just say that "I don't know" and do NOT add any text or explain anything after this but don't make up an answer on your own.
    3. If you feel like the given task cannot be accomplished with the given context, just say that "I don't know" and do NOT add any text or explain anything after this but do not assume information on your own. Some examples of this can be "Summarize this pdf" and such tasks that require entire pdf and not just some extracted context.
    4. Keep the answer concise and to the point.

    Context: {context}

    Question: {question}

    Helpful Answer:"""


    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt) 

    # Define llm
    llm = Ollama(model="llama3.1:8b")

    llm_chain = LLMChain(
        llm=llm, 
        prompt=QA_CHAIN_PROMPT, 
        callbacks=None, 
        # verbose=True,
    )

    document_prompt = PromptTemplate(
        input_variables=["page_content"],
        template="Context:\ncontent:{page_content}",
    )

    combine_documents_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context",
        document_prompt=document_prompt,
        callbacks=None,
    )

    qa = RetrievalQA(
        combine_documents_chain=combine_documents_chain,
        # verbose=True,
        retriever=retriever,
        return_source_documents=True,
    )

    return qa.invoke(question)["result"]