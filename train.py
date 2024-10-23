# Imports
import os
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import torch
import logging
import transformers

from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Set up logging
logging.basicConfig(
    filename="training_log.txt",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


# Settings
model_checkpoint = "Helsinki-NLP/opus-mt-en-fr"
max_length = 512
source_lang = "en"
target_lang = "fr"

# Define a callback to log the loss
class LoggingCallback(transformers.TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            logging.info(logs)



dataset = load_dataset("csv", data_files={
    "train": "train.csv",
    "validation": "validation.csv",
    "test": "test.csv"
})

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, return_tensors="pt")

def preprocess_function(examples):
    inputs = [ex for ex in examples["source"]]
    targets = [ex for ex in examples["target"]]
    model_inputs = tokenizer(
        inputs, text_target=targets, max_length=max_length, truncation=True
    )
    return model_inputs

tokenized_datasets = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
)

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint) 

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="pt")

model_args = Seq2SeqTrainingArguments(
    f"saved_model/{model_checkpoint}-finetuned-{source_lang}-to-{target_lang}",
    evaluation_strategy="steps",
    learning_rate=2e-4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.02,
    save_total_limit=3,
    max_steps=600000,  # Number of total training steps
    warmup_steps=500,  # Number of warmup steps
    predict_with_generate=True,
    push_to_hub=False,
    report_to="none",
    eval_steps=2500,  # Evaluate every 500 steps to log validation loss
    logging_steps=2500,  # Log training stats every 500 steps
    metric_for_best_model="eval_loss",
    greater_is_better=False,  # Lower validation loss is better
)

trainer = Seq2SeqTrainer(
    model=model,
    args=model_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    callbacks=[LoggingCallback()]
)

trainer.train()

