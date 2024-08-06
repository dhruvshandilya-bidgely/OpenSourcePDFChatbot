import langchain_core.documents.base
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.documents import Document
from langchain.chains import StuffDocumentsChain, ReduceDocumentsChain, MapReduceDocumentsChain
from utils.clear_yes_and_no import clear_yes_no_folder

# Class to setup output format of answer_bot.
class AnswerWithBoolFlag(BaseModel):
    answer: str = Field(description="Answer in accordance to the context and question. Make sure to use information from the context only. If context is not sufficient for answering the given question just say 'I do not know.' but do not make assumptions.")
    relevance: str = Field(description="YES if the question can be answered while strictly using only the context. NO otherwise.")


def answer_bot_with_bool_flag(context: langchain_core.documents.base.Document, question: str):
    """
    Verifies if context holds information relevant to the question.

    Args:
        context: (langchain_core.documents.base.Document) The langchain doc we want to verify.
        question: (String) The question we are interested in.

    Returns:
        answer_dict : (dict) Is a dict containing the 'answer' and 'relevance' i.e. whether the LLM thinks answer can be obtained from text or not.
    """
    # Using LLaMa 3.1 using ollama
    llm = Ollama(model="llama3.1:8b", temperature=0)

    # Set up a parser + inject instructions into the prompt template.
    parser = JsonOutputParser(pydantic_object=AnswerWithBoolFlag)

    classif_prompt = """You are a helpful assistant.
    {format_instructions}
    You are given a 'Context' and 'Question' below both delimeted by ```. 
    The context you are given is a chunk or part of a larger document. 
    
    Context: ```{context}```

    Question: ```{question}```
    """

    prompt = PromptTemplate(
        template=classif_prompt,
        input_variables=["context","question"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser

    try:
        result_dict = chain.invoke({"context": context.page_content,"question":question})
    except:
        return ""
    return result_dict


def map_reduce(docs: list[langchain_core.documents.base.Document], question: str) -> str:
    """
    Uses map reduce to try to answer our question.

    Args:
        docs: (list[langchain_core.documents.base.Document]) Is a list of langchain documents of our pdf.
        question: (String) The question we are interested in.

    Returns:
        final_answer : (string) Is a string containing the final answer that the LLM creates using map reduce.
    """

    map_prompt = """Use the following context below to answer the question at the end.

    Context: {context}

    Question: {question}

    ANSWER: 
    """
    print('Entered map reduce.')

    map_prompt_template = PromptTemplate(template=map_prompt, 
        input_variables=["context"],
        partial_variables={"question":question})
    
    # Using LLaMa 3.1 using ollama
    llm = Ollama(model="llama3.1:8b", temperature=0)

    map_chain = LLMChain(llm=llm,prompt=map_prompt_template)

    reduce_prompt = """Write a answer to the question given below using the following text.
    Try to combine the texts given below to form a comprehensive and consistent answer.
    If you feel you cannot answer using given text just say that "I don't know" but don't make up an answer on your own.

    text: {text}

    Question: {question}

    ANSWER: 
    """

    reduce_prompt_template = PromptTemplate(template=reduce_prompt, 
        input_variables=["text"],
        partial_variables={"question":question})

    reduce_chain = LLMChain(llm=llm,prompt=reduce_prompt_template)

    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="text", verbose = True
    )

    reduce_documents_chain = ReduceDocumentsChain(
        combine_documents_chain=combine_documents_chain,
        collapse_documents_chain=combine_documents_chain,
        token_max=4000, verbose = True)

    map_reduce_chain = MapReduceDocumentsChain(
        llm_chain=map_chain,
        reduce_documents_chain=reduce_documents_chain,
        document_variable_name="context",
        return_intermediate_steps=True, verbose = True)
    
    output = map_reduce_chain.invoke(docs)

    return output["output_text"]


def final_step(docs: list[langchain_core.documents.base.Document], question: str) -> str:
    """
    Generates final answer to the question.

    Args:
        docs: (list[langchain_core.documents.base.Document]) Is a list of langchain documents of our pdf.
        question: (String) The question we are interested in.

    Returns:
        final_answer : (string) Is a string containing the final answer that the LLM thinks is appropriate.
    """

    selected_docs = []
    cnt = 0

    clear_yes_no_folder()

    for i in range(len(docs)):
        for j in range(2):
            result_dict = answer_bot_with_bool_flag(docs[i],question)
            try:
                if result_dict['relevance']=='YES':
                    document = Document(page_content=docs[i].page_content,source=f'doc number {i}')
                    selected_docs.append(document)
                    cnt += 1
                    print(f"YES for {i}")
                    f = open(f"DEBUG/YES/{i}.txt", "w")
                    f.write(f"Reasoning : {result_dict['answer']}")
                    f.close()
                    break
                else:
                    print(f"NO for {i}")
                    f = open(f"DEBUG/NO/{i}.txt", "w")
                    f.write(f"Reasoning : {result_dict['answer']}")
                    f.close()
                    break
            
            except KeyError as e:
                print(f'Exception in doc {i}, relevance key does not exist.')
                print(result_dict)
                print(e)
                continue
            
            except Exception as e:
                print(f'Exception in doc {i}')
                print(result_dict)
                print(e)
                continue


    print("Selection fraction --> ",(cnt/len(docs)))

    output = map_reduce(selected_docs,question)
    
    return output