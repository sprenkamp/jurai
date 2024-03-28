import sys
sys.path.append("src/langchain_agent/")

from chroma_retrieve import retrieve_chromadb
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage
import langdetect

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

lang_dict = {
    "en": "english",
    "de": "german",
    "fr": "french",
    "it": "italian",
    'uk': 'ukrainian',
    'ru': 'russian',
}

vectordb = retrieve_chromadb(collection_name="swiss_refugee_info")

def chat(message: str, chat_history: list) -> str:
    message_language = detect_language(message)
    if message_language in lang_dict.keys():
        message_language = lang_dict[message_language]
    else:
        message_language = "english"
    if message_language != "english":
        message = translate_to_english(message)
    check = check_message_content(message)
    print(check)
    if check == True:
        result = run_conversational_retrieval_chain(message, chat_history, message_language)
        response=result['answer']
        rephrased_question=result["generated_question"]
        top_k_docs=result['source_documents']
    else:
        response = run_normal_chain(message, message_language)
        top_k_docs = None
        rephrased_question = None

    return response, rephrased_question, top_k_docs 
        
def detect_language(message: str) -> str:
    return langdetect.detect(message)

def translate_to_english(message: str) -> str:
    prompt = [
            SystemMessage(
                content=f"Please translate the following message to English"
            ),
            HumanMessage(
                content=f"{message}"
            ),
        ]
    output = llm(prompt)
    return output.content

def check_message_content(message: str) -> bool:
    prompt = [
            SystemMessage(
                # content=f"""Determine if a message pertains to refugee and asylum matters:
                # 0 for unrelated content (e.g., "Hi, how are you?", "Best restaurants in Paris?")
                # 1 for content related to:
                #     - Asylum or protection statuses
                #     - Refugee Entry/Exit procedures
                #     - Accommodations and allocation processes
                #     - Medical care and humanitarian support
                #     - Employment, education, and financial matters, subsidies, and support
                #     - humanitarian efforts and concerns
                #     (e.g. "How can refugees enter a country?" or "What is a specific protection status?")
                # You SOLEY return 0 or 1"""
                content=f"""Determine if a personal or refugee or private matters:
                0 for unrelated personal content (e.g., "Hi, how are you?", "Best restaurants in Paris?")
                1 for content related to refugee matters and
                You SOLEY return 0 or 1"""
            ),
            HumanMessage(
                content=f"{message}"
            ),
        ]
    output = llm(prompt)
    return output.content == "1" #will return True if message is related to refugee-related topics and False if not

def run_conversational_retrieval_chain(message: str, chat_history: list, message_language: str) -> str:
    # Build prompt
    prompt_template = """Use the following pieces of context to answer the question at the end. Combine the information from the context with your own general knowledge to provide a comprehensive and accurate answer. Please be as specific as possible. Also answer with the language the question is phrased in.
    {context}
    Question: {question}
    Helpful Answer in {message_language} language:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template)

    qa_ConversationalRetrievalChain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={'k': 5, 'lambda_mult': 0.25}),# Retrieve more documents with higher diversity- useful if your dataset has many similar documents
        return_source_documents=True,
        return_generated_question=True,
        combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    result_ConversationalRetrievalChain = qa_ConversationalRetrievalChain({"question": message, "chat_history": chat_history, "message_language": message_language})
    # chat_history.append((message, result_ConversationalRetrievalChain["answer"]))

    return result_ConversationalRetrievalChain

def run_normal_chain(message: str, message_language: str) -> str:
    prompt = [
            SystemMessage(
                content=f"Be a helpful assistant and answer in {message_language}"
            ),
            HumanMessage(
                content=f"{message}"
            ),
        ]
    output = llm(prompt)
    return output.content


