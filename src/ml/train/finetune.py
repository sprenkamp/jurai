from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json 

train_dataset = load_dataset('json', data_files='notes.jsonl', split='train')
eval_dataset = load_dataset('json', data_files='notes_validation.jsonl', split='train')

class Finetune:
    def __init__(self, model_config_path) -> None:
        self.model_config = self.load_config(model_config_path)
        self.base_model_id = model_config_path['base_model_id']
        self.data_path = model_config_path['data_path']
        self.model, self.tokenizer = self.load_model_and_tokenizer()

    def load_config(self, model_config_path):
        with open(model_config_path, 'r') as f:
            model_config = json.load(f)
        return model_config

    def load_model_and_tokenizer(self):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(self.base_model_id,
                                                      quantization_config=bnb_config) 

        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_id,
            padding_side="left",
            add_eos_token=True,
            add_bos_token=True,
        )
        tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer
    
    def load_and_preprocess(self):
        train_dataset = load_dataset('json', data_files='notes.jsonl', split='train')
        val_dataset = load_dataset('json', data_files='notes_validation.jsonl', split='train')


    