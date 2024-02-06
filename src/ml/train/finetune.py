import os
import yaml
from datetime import datetime
import torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
                          DataCollatorForLanguageModeling, BitsAndBytesConfig)
from peft import LoraConfig, get_peft_model
import transformers
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import (FullOptimStateDictConfig,
                                                                FullStateDictConfig)

class Finetune:
    """A class for fine-tuning pre-trained causal language models with specific configurations
    for quantization and parameter-efficient fine-tuning techniques."""

    def __init__(self, model_config_path: str) -> None:
        """Initializes the Finetune class with model configuration.

        Args:
            model_config_path (str): Path to the JSON file containing model configuration.
        """
        self.model_config = self.load_config(model_config_path)
        self.model, self.tokenizer = self.load_model_and_tokenizer()

    def load_config(self, model_config_path: str) -> dict:
        """Loads model configuration from a YAML file.

        Args:
            model_config_path (str): Path to the model configuration file.

        Returns:
            dict: The model configuration as a dictionary.
        """
        with open(model_config_path, 'r') as f:
            model_config = yaml.safe_load(f)
        return model_config

    def load_model_and_tokenizer(self) -> tuple:
        """Loads the pre-trained model and tokenizer based on the configuration.

        Returns:
            tuple: A tuple containing the model and tokenizer instances.
        """
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(self.model_config['base_model_id'],
                                                     quantization_config=bnb_config)
        tokenizer = AutoTokenizer.from_pretrained(self.model_config['base_model_id'],
                                                   padding_side="left",
                                                   add_eos_token=True,
                                                   add_bos_token=True)
        tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer

    def load_and_tokenize(self) -> None:
        """Loads and tokenizes the training and validation datasets."""
        self.train_dataset = load_dataset('json', data_files=self.model_config['train_path'], split='train')
        self.val_dataset = load_dataset('json', data_files=self.model_config['val_path'], split='train') if 'val_data_file' in self.model_config else None

        def tokenize_map_function(examples):
            """Tokenizes examples for processing.

            Args:
                examples: A batch from the dataset.

            Returns:
                Tokenized examples.
            """
            return self.tokenizer(examples["text"], truncation=True, max_length=self.model_config['block_size'], padding="max_length")

        self.tokenized_train_dataset = self.train_dataset.map(tokenize_map_function, batched=True)
        if self.val_dataset is not None:
            self.tokenized_val_dataset = self.val_dataset.map(tokenize_map_function, batched=True)

    def wandb_init(self) -> None:
        """Initializes Weights & Biases for experiment tracking."""
        import wandb
        wandb.login()
        os.environ["WANDB_PROJECT"] = self.model_config.get('wandb_project', 'journal-finetune')

    def peft_and_accelerator(self) -> None:
        """Prepares the model for parameter-efficient fine-tuning and sets up the accelerator for distributed training."""
        config = LoraConfig(
            r=self.model_config["lora_r"],
            lora_alpha=self.model_config["lora_alpha"],
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
            bias="none",
            lora_dropout=self.model_config["lora_dropout"],
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, config)

        fsdp_plugin = FullyShardedDataParallelPlugin(
            state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
            optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
        )
        accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
        self.model, self.tokenizer, self.tokenized_train_dataset, self.tokenized_val_dataset = accelerator.prepare(self.model, self.tokenizer, self.tokenized_train_dataset, self.tokenized_val_dataset)

    def train(self):
        """Conducts the training process."""
        
        self.peft_and_accelerator()
        if self.model_config["wandb"]:
            self.wandb_init()

        project = "journal-finetune"
        base_model_name = "mistral"
        run_name = base_model_name + "-" + project
        output_dir = "./" + run_name

        trainer = transformers.Trainer(
            model=self.model,
            train_dataset=self.tokenized_train_dataset,
            eval_dataset=self.tokenized_val_dataset,
            args=transformers.TrainingArguments(
                output_dir=output_dir,
                warmup_steps=1,
                per_device_train_batch_size=int(self.model_config["batch_size"]),
                gradient_checkpointing=True,
                max_steps=self.model_config["max_steps"],
                learning_rate=self.model_config["learning_rate"],
                warmup_ratio=self.model_config["warmup_ratio"],
                weight_decay=self.model_config["weight_decay"],
                gradient_accumulation_steps=int(self.model_config["gradient_accumulation_steps"]),
                fp16=self.model_config["fp16"], 
                optim="paged_adamw_8bit",
                logging_steps=25,              # When to start reporting loss
                logging_dir="./logs",        # Directory for storing logs
                save_strategy="steps",       # Save the model checkpoint every logging step
                save_steps=25,                # Save checkpoints every 50 steps
                evaluation_strategy="steps", # Evaluate the model every logging step
                eval_steps=25,               # Evaluate and save checkpoints every 50 steps
                do_eval = True if self.val_dataset is not None else False,
                report_to = self.model_config["wandb"] if self.model_config["wandb"] else False,
                run_name = f"{self.model_config['wandb']}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}" if self.model_config["wandb"] else None,
                push_to_hub=self.model_config["push_to_hub"],
            ),
            data_collator=transformers.DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
        )

        self.model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
        trainer.train()
        self.model.save_pretrained(output_dir)

if __name__ == "__main__":
    import argparse
    # Parse command-line arguments for model configuration path
    parser = argparse.ArgumentParser(description="Fine-tune a pre-trained language model.")
    parser.add_argument('--config', type=str, help='Path to the model config file')
    args = parser.parse_args()
    # Initialize and run the fine-tuning process
    print(args.config)
    finetune = Finetune(args.config)
    finetune.load_and_tokenize()
    finetune.train()