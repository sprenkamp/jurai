import json
import os
import yaml
import csv
import argparse
from tqdm import tqdm
from langchain.schema import HumanMessage, SystemMessage
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from src.rag.chroma_db.chroma_retrieve import retrieve_chromadb
# from langchain.chains.retrieval_qa.base import BaseRetrievalQA, RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

class Testing:
    def __init__(self, model_config_path) -> None:
        self.model_config = self.load_config(model_config_path)
        self.load_model()
        if self.model_config["rag"]:
            self.chromadb = retrieve_chromadb(collection_name="jurai_laws_gg")

    def load_config(self, model_config_path):
        with open(model_config_path, 'r') as f:
            model_config = yaml.load(f, Loader=yaml.FullLoader)
        return model_config

    def load_model(self):
        print("using", self.model_config["repo_id"], "for evaluation")
        if "gpt" in self.model_config["repo_id"]:
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
            temperature=0,  # Make as deterministic as possible
            model_name=self.model_config['repo_id'],
            max_tokens=4096,
        )
        else:
            from huggingface_hub import InferenceClient
            # HF Inference Endpoints parameter
            endpoint_url = self.model_config["inference_endpoint"]
            hf_token = os.environ.get("HF_TOKEN")
            # generation parameter
            self.gen_kwargs = dict(
                max_new_tokens=488,
                top_k=30,
                top_p=0.9,
                temperature=0.1,
                repetition_penalty=1.02,
                stop_sequences=["\nUser:", "<|endoftext|>", "</s>"],
            )

            # Streaming Client
            self.llm = InferenceClient(endpoint_url, token=hf_token)


    def test(self):
        splits = self.model_config["repo_id"].split("/")
        if len(splits) > 1:
            model_name = splits[1]
        else:
            model_name = splits[0]
        out_path = self.model_config["test_result_path"] + model_name + "_" + self.model_config["test_path"].split("/")[-1].replace(".txt", ".csv")
        with open(self.model_config["test_path"], 'r') as input_file, open(out_path, 'w', newline='') as output_file:
            questions_from_input, answers_from_input = zip(*[(row[0], row[1]) for row in csv.reader(input_file, delimiter=';')][1:])
            csv_writer = csv.writer(output_file)
            csv_writer.writerow(["Question", "Answer"])  # Writing the header row
            for question in tqdm(questions_from_input):
                if question:
                    if "gpt" in self.model_config["repo_id"]:
                        if not self.model_config["rag"]:
                            prompt= [
                                    SystemMessage(
                                        content="Du bist deutscher Rechtsexperte, der Gesetze und Vorschriften anwendet und interpretiert und Rechtsfragen löst. Du sollst klare und ausführliche Antworten geben und die einschlägigen Gesetze zitieren. Wenn dir eine Frage unklar oder mehrdeutig gestellt wird, sollst du darauf hinweisen und Vorschläge zur Klärung anbieten. Wenn du keine klare Antwort geben kannst, weil die Rechtslage unklar oder mehrdeutig ist, dann sollst du darauf hinweisen und die verschiedenen Lösungsmöglichkeiten abbilden."
                                    ),
                                    HumanMessage(
                                        content=question
                                    ),
                                ]
                            output_llm = self.llm(prompt)
                            csv_writer.writerow([question, output_llm.content])
                        else:
                            # k=50 #allow for more documents to be retrieved for 8k tokens of gpt-4
                            prompt_template = """Du bist deutscher Rechtsexperte, der Gesetze und Vorschriften anwendet und interpretiert und Rechtsfragen löst. Du sollst klare und ausführliche Antworten geben und die einschlägigen Gesetze zitieren. Wenn dir eine Frage unklar oder mehrdeutig gestellt wird, sollst du darauf hinweisen und Vorschläge zur Klärung anbieten. Wenn du keine klare Antwort geben kannst, weil die Rechtslage unklar oder mehrdeutig ist, dann sollst du darauf hinweisen und die verschiedenen Lösungsmöglichkeiten abbilden. Um die Frage von Nutzern zu beantworten musst du auf folgende Dokumente zugreifen und auf nix anderes die Dokumente sind das einzige Gesetz was zählt (e.g. Gesetze, Verordnungen, Urteile, Kommentare, etc.) zugreifen.  
                            {context}
                            Frage: {question}"""
                            QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template)
                            qa_ConversationalRetrievalChain = RetrievalQA.from_chain_type(
                                llm=self.llm,
                                retriever=self.chromadb.as_retriever(search_type="mmr", search_kwargs={'k': 5,}),
                                                                                                        #'lambda_mult': 0.25}),
                                return_source_documents=True,
                                # return_generated_question=True,
                                chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
                            )
                        result_ConversationalRetrievalChain = qa_ConversationalRetrievalChain({"query": question})
                        print(result_ConversationalRetrievalChain)
                    else:
                        output_llm = self.llm.text_generation(question, stream=False, details=True, **self.gen_kwargs)
                        csv_writer.writerow([question, output_llm.generated_text])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test a model.')
    parser.add_argument('--config_path', type=str, help='Path to the model configuration file.')
    args = parser.parse_args()
    testing = Testing(args.config_path)
    testing.test()