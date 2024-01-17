import json
import yaml 
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms.huggingface_pipeline import HuggingFacePipeline



class Finetune:
    def __init__(self, model_config_path) -> None:
        self.model_config = self.load_config(model_config_path)

    def load_config(self, model_config_path):
        with open(model_config_path, 'r') as f:
            model_config = yaml.load(f, Loader=yaml.FullLoader)
        return model_config

    def load_model_and_tokenizer_hf(self):
        print(self.model_config["repo_id"])
        tokenizer = AutoTokenizer.from_pretrained(self.model_config["repo_id"])
        model = AutoModelForCausalLM.from_pretrained(self.model_config["repo_id"])
        pipe = pipeline("text-generation", 
               model=model, 
               tokenizer=tokenizer, 
               max_new_tokens=512
               )

        self.llm = HuggingFacePipeline(pipeline=pipe)



if __name__ == "__main__":
    model_config_path = "src/ml/config/bloke_german.yaml"
    finetune = Finetune(model_config_path)
    finetune.load_model_and_tokenizer_hf()
    finetune.llm("Hello world")