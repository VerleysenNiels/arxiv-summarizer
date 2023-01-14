"""
Main script for finetuning a transformer model on a dataset. In this case I will be training a DistilBart model on an arXiv summarization dataset.
This script will be running on CUDA with PyTorch as deep learning framework.    
"""

import logging
logging.basicConfig(level=logging.INFO)

import nltk

# Import pytorch in order to use gpu
import torch

# To download and manage the dataset from HuggingFace
from datasets import load_dataset

# Bart tokenizer and model classes
from transformers import BartTokenizer, BartForConditionalGeneration

# To make training easier, the transformers library comes with some extra tools
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq

# Same for evaluation
import evaluate


if __name__ == "__main__":    
    # Can we use gpu?
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Running on {device}")
    
    logging.info("Loading dataset...")
    # Load the dataset
    arxiv_dataset = load_dataset("ccdv/arxiv-summarization", cache_dir="cache")
    # We need to select the text and summary columns as the dataset can be used for other things as well
    text_column = "article"
    summary_column = "abstract" 
    
    logging.info("Loading model and tokenizer...")
    # We can automatically load in the tokenizer and model that we want to use
    distilbart_tokenizer = BartTokenizer.from_pretrained("sshleifer/distilbart-cnn-6-6", cache_dir="cache")
    distilbart_model = BartForConditionalGeneration.from_pretrained("sshleifer/distilbart-cnn-6-6", cache_dir="cache")
    distilbart_model.to(device)
    
    logging.info("Preparing dataset...")
    # We can then tokenize our dataset
    def tokenize_function(samples):
        articles = distilbart_tokenizer(samples[text_column], truncation=True)
        abstracts = distilbart_tokenizer(samples[summary_column], truncation=True)
        
        articles["labels"] = abstracts["input_ids"]
        return articles
    
    tokenized_arxiv_dataset = arxiv_dataset.map(tokenize_function, batched=True)
    
    # Initialize a collator for automatic padding of inputs and targets
    data_collator = DataCollatorForSeq2Seq(distilbart_tokenizer, model=distilbart_model)    
    
    logging.info("Preparing trainer...")
    # Next up let's define our training arguments
    training_arguments = Seq2SeqTrainingArguments(
        output_dir="distilbart-6-6-arxiv",  # Where to store results
        overwrite_output_dir=True,          # Overwrite the content of the output directory
        evaluation_strategy="epoch",        # Evaluate the model every epoch
        save_strategy="epoch",              # Saving strategy should be the same as the evaluation strategy
        num_train_epochs=3,                 # The number of training epochs
        per_device_train_batch_size=8,      # Batch size for training
        per_device_eval_batch_size=2,       # Batch size for evaluation
        learning_rate=1e-5,                 # Set the learning rate for finetuning
        load_best_model_at_end=True,        # When training is done, load the best performing checkpoint
        report_to='tensorboard'             # Always nice to be able to follow the training progress remotely :)       
    )
    
    # And of course the evaluation metric, as the pretrained distilbart was evaluated with the rouge metric, I will use it here again
    metric = evaluate.load("rouge")
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        
        # Decode the generated summaries and the target summaries
        decoded_preds = distilbart_tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = distilbart_tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # In order to use rouge, we need to add a newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
        
        # Compute rouge scores
        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        
        return {k: round(v, 4) for k, v in result.items()}
    
    # Finally we can set up the trainer
    trainer = Seq2SeqTrainer(
        distilbart_model,
        training_arguments,
        train_dataset=tokenized_arxiv_dataset["train"],
        eval_dataset=tokenized_arxiv_dataset["validation"],
        data_collator=data_collator,
        tokenizer=distilbart_tokenizer,
        compute_metrics=compute_metrics
    )
    
    # And we can run our trainer
    logging.info("Started training...")
    trainer.train()
    
    
    
    
