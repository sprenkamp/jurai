import json
import yaml
import csv
import argparse
from tqdm import tqdm
from langchain.schema import HumanMessage, SystemMessage
from langchain.llms.huggingface_pipeline import HuggingFacePipeline



class Testing:
    def __init__(self, model_config_path) -> None:
        self.model_config = self.load_config(model_config_path)
        self.load_model()

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
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            tokenizer = AutoTokenizer.from_pretrained(self.model_config["repo_id"])
            model = AutoModelForCausalLM.from_pretrained(self.model_config["repo_id"])
            pipe = pipeline("text-generation", 
                model=model, 
                tokenizer=tokenizer, 
                max_new_tokens=4096,
                )
            self.llm = HuggingFacePipeline(pipeline=pipe)

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
                        from langchain.prompts import PromptTemplate
                        template = """Du bist deutscher Rechtsexperte, der Gesetze und Vorschriften anwendet und interpretiert und Rechtsfragen löst. Du sollst klare und ausführliche Antworten geben und die einschlägigen Gesetze zitieren. Wenn dir eine Frage unklar oder mehrdeutig gestellt wird, sollst du darauf hinweisen und Vorschläge zur Klärung anbieten. Wenn du keine klare Antwort geben kannst, weil die Rechtslage unklar oder mehrdeutig ist, dann sollst du darauf hinweisen und die verschiedenen Lösungsmöglichkeiten abbilden.
                        Frage: {question}
                        Antwort: """
                        prompt = PromptTemplate.from_template(template)

                        chain = prompt | self.llm

                        print(chain)

                        print(chain.invoke({"question": question}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test a model.')
    parser.add_argument('--config_path', type=str, help='Path to the model configuration file.')
    args = parser.parse_args()

    testing = Testing(args.config_path)
    testing.test()