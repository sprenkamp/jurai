import json
import yaml
import csv
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chat_models import ChatOpenAI



class Evaluation:
    def __init__(self, model_config_path) -> None:
        self.model_config = self.load_config(model_config_path)

    def load_config(self, model_config_path):
        with open(model_config_path, 'r') as f:
            model_config = yaml.load(f, Loader=yaml.FullLoader)
        return model_config

    def load_model(self):
        print("using", self.model_config["repo_id"], "for evaluation")
        if "gpt" in self.model_config["repo_id"]:
            self.llm = ChatOpenAI(
            temperature=0,  # Make as deterministic as possible
            model_name=self.model_config['model_name'],
        )
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model_config["repo_id"])
            model = AutoModelForCausalLM.from_pretrained(self.model_config["repo_id"])
            pipe = pipeline("text-generation", 
                model=model, 
                tokenizer=tokenizer, 
                max_new_tokens=512
                )
            self.llm = HuggingFacePipeline(pipeline=pipe)

    def eval(self):
        splits = self.model_config["repo_id"]
        if len(splits) > 1:
            model_name = splits[1]
        else:
            model_name = splits[0]
        out_path = self.model_config["eval_result_path"] + "_" + model_name + "_" + self.model_config["eval_path"].split("/")[-1].replace(".txt", ".csv")
        with open(self.model_config["eval_path"], 'r') as input_file, open(out_path, 'w', newline='') as output_file:
            csv_writer = csv.writer(output_file)
            csv_writer.writerow(["Question", "Answer"])  # Writing the header row
            for line in input_file:
                line = line.strip()  # Remove any leading/trailing whitespace
                if line:
                    answer = self.llm(line)
                    csv_writer.writerow([line, answer[0]["generated_text"]])

if __name__ == "__main__":
    model_config_path = "src/ml/config/bloke_german.yaml"
    model_config_path = "src/ml/config/bloke_german.yaml"
    evaluation = Evaluation(model_config_path)
    evaluation.load_model()
    evaluation.eval()