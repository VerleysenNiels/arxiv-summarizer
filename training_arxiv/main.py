"""
Main script for finetuning a transformer model on a dataset. In this case I will be training a Longformer Encoder-Decoder model (LED) on an arXiv summarization dataset.
LED allows for much longer token sequences as inputs than for instance distilbart. We need this as scientific papers are longer pieces of text.
This script will be running on CUDA with PyTorch as deep learning framework.
"""

import logging
logging.basicConfig(filename="logs/train.log", level=logging.INFO)

import numpy as np

import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Import pytorch in order to use gpu
import torch

# To download and manage the dataset from HuggingFace
from datasets import load_dataset

# Bart tokenizer and model classes
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# To make training easier, the transformers library comes with some extra tools
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

# Same for evaluation
import evaluate

# Let's first put some limits on the sequence lengths
MAX_INPUT_SEQUENCE_LENGTH = 10240   # Average is around 6k tokens
MAX_OUTPUT_SEQUENCE_LENGTH = 600    # Average is around 300 tokens

# We need the batch size in a number of places
BATCH_SIZE = 4

if __name__ == "__main__":
    # Can we use gpu?
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Running on {device}")
    
    logging.info("Loading dataset...")
    # Load the dataset
    train_arxiv_dataset = load_dataset("ccdv/arxiv-summarization", split="train", cache_dir="cache")
    eval_arxiv_dataset = load_dataset("ccdv/arxiv-summarization", split="validation", cache_dir="cache")

    logging.info("Loading model and tokenizer...")
    # We can automatically load in the tokenizer and model that we want to use
    tokenizer = AutoTokenizer.from_pretrained("allenai/led-base-16384", cache_dir="cache")
    led_model = AutoModelForSeq2SeqLM.from_pretrained("allenai/led-base-16384", gradient_checkpointing=True, cache_dir="cache")
    
    # We need to configure the hyperparameters of the model
    # I'm here starting from standard settings
    led_model.config.num_beams = 2
    led_model.config.max_length = MAX_OUTPUT_SEQUENCE_LENGTH
    led_model.config.min_length = 100
    led_model.config.length_penalty = 2.0
    led_model.config.early_stopping = True
    led_model.config.no_repeat_ngram_size = 3
    
    # And load the model into the GPUs
    led_model.to(device)

    logging.info("Preparing dataset...")
    # We can then tokenize our dataset
    def tokenize_function(batch):
        # tokenize the inputs and labels
        inputs = tokenizer(
            batch["article"],
            padding="max_length",
            truncation=True,
            max_length=MAX_INPUT_SEQUENCE_LENGTH,
        )
        
        outputs = tokenizer(
            batch["abstract"],
            padding="max_length",
            truncation=True,
            max_length=MAX_OUTPUT_SEQUENCE_LENGTH,
        )

        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask

        # Prepare the global attention mask, with zeroes
        batch["global_attention_mask"] = len(batch["input_ids"]) * [[0 for _ in range(len(batch["input_ids"][0]))]]

        # Update the 0 index for all samples
        batch["global_attention_mask"][0][0] = 1
        batch["labels"] = outputs.input_ids

        # PAD tokens need to be ignored
        batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]

        return batch

    tokenized_train_arxiv_dataset = train_arxiv_dataset.map(tokenize_function, batched=True, batch_size=BATCH_SIZE)
    tokenized_eval_arxiv_dataset = eval_arxiv_dataset.map(tokenize_function, batched=True, batch_size=BATCH_SIZE)
    
    # Set the correct format of the tokenized dataset
    tokenized_train_arxiv_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
    )
    tokenized_eval_arxiv_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
    )

    logging.info("Preparing trainer...")
    # Next up let's define our training arguments
    training_arguments = Seq2SeqTrainingArguments(
        output_dir="distilbart-6-6-reddit",     # Where to store results
        overwrite_output_dir=True,              # Overwrite the content of the output directory
        evaluation_strategy="steps",            # Evaluate the model every epoch
        eval_steps=10000,                       # Evaluate the model every 10k update steps
        eval_accumulation_steps=1,              # Reduce memory usage during validation (keep low to not overflow RAM)
        #fp16=True,                              # Reduce memory consumption by working with 16-bit floating point numbers instead of 32-bit
        num_train_epochs=3,                     # The number of training epochs
        per_device_train_batch_size=BATCH_SIZE, # Batch size for training
        per_device_eval_batch_size=1,           # Batch size for evaluation
        learning_rate=1e-5,                     # Set the learning rate for finetuning
        predict_with_generate=True,             # Important for evaluation
        save_total_limit = 1,                   # Minimize storage usage
        save_strategy="steps",                  # Save best model every 10k update steps
        save_steps=10000,                       # Save best model every 10k update steps
        report_to='tensorboard',                # Always nice for following up on the training process
        push_to_hub=True                        # Push the results to the hub
    )

    # And of course the evaluation metric
    metric = evaluate.load("rouge")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        # Decode the generated summaries and the target summaries
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # In order to use rouge, we need to add a newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

        # Compute rouge scores
        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

        return {k: round(v, 4) for k, v in result.items()}

    # Finally we can set up the trainer
    trainer = Seq2SeqTrainer(
        led_model,
        training_arguments,
        train_dataset=tokenized_train_arxiv_dataset,
        eval_dataset=tokenized_eval_arxiv_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # And we can run our trainer
    logging.info("Started training...")
    result = trainer.train()
    logging.info("Done")
    logging.info(f"Time: {result.metrics['train_runtime']:.2f}")
    logging.info(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")

    logging.info(f"Pushing to hub...")
    trainer.push_to_hub()
    
    
    
    
