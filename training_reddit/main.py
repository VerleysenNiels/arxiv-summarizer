"""
Main script for finetuning a transformer model on a dataset. In this case I will be training a DistilBart model on a reddit summarization dataset.
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
    reddit_dataset = load_dataset("reddit", cache_dir="cache")

    # We need to select the text and summary columns as the dataset can be used for other things as well
    text_column = "content"
    summary_column = "summary"

    logging.info("Loading model and tokenizer...")
    # We can automatically load in the tokenizer and model that we want to use
    distilbart_tokenizer = BartTokenizer.from_pretrained("sshleifer/distilbart-cnn-6-6", cache_dir="cache")
    distilbart_model = BartForConditionalGeneration.from_pretrained("sshleifer/distilbart-cnn-6-6", cache_dir="cache")
    distilbart_model.to(device)

    logging.info("Preparing dataset...")
    # We can then tokenize our dataset
    def tokenize_function(samples):
        reddit_posts = distilbart_tokenizer(samples[text_column], truncation=True)
        summaries = distilbart_tokenizer(samples[summary_column], truncation=True)

        reddit_posts["labels"] = summaries["input_ids"]
        return reddit_posts

    tokenized_reddit_dataset = reddit_dataset.map(tokenize_function, batched=True)

    # Split the dataset as there is no split in yet
    reddit_dataset = tokenized_reddit_dataset['train'].train_test_split(test_size=0.01)
    train_reddit_dataset = reddit_dataset['train']
    val_reddit_dataset = reddit_dataset['test']

    # Initialize a collator for automatic padding of inputs and targets
    data_collator = DataCollatorForSeq2Seq(distilbart_tokenizer, model=distilbart_model)

    logging.info("Preparing trainer...")
    # Next up let's define our training arguments
    training_arguments = Seq2SeqTrainingArguments(
        output_dir="distilbart-cnn-6-6-reddit",     # Where to store results
        overwrite_output_dir=True,                  # Overwrite the content of the output directory
        evaluation_strategy="epoch",                # Evaluate the model every epoch
        eval_accumulation_steps=1,                  # Reduce memory usage during validation (keep low to not overflow RAM)
        num_train_epochs=3,                         # The number of training epochs
        per_device_train_batch_size=8,              # Batch size for training
        per_device_eval_batch_size=8,               # Batch size for evaluation
        learning_rate=1e-4,                         # Set the learning rate for finetuning
        predict_with_generate=True,                 # Important for evaluation
        save_total_limit = 1,                       # Minimize storage usage
        save_strategy="steps",                      # Save best model every 10k update steps
        save_steps=10000,                           # Save best model every 10k update steps
        report_to='tensorboard',                    # Always nice for following up on the training process
        push_to_hub=True                            # Push the results to the hub
    )

    # And of course the evaluation metric
    metric = evaluate.load("rouge")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        # Decode the generated summaries and the target summaries
        decoded_preds = distilbart_tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, distilbart_tokenizer.pad_token_id)
        decoded_labels = distilbart_tokenizer.batch_decode(labels, skip_special_tokens=True)

        # In order to use rouge, we need to add a newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

        # Compute rouge scores
        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        #result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        return {k: round(v, 4) for k, v in result.items()}

    # Finally we can set up the trainer
    trainer = Seq2SeqTrainer(
        distilbart_model,
        training_arguments,
        train_dataset=train_reddit_dataset,
        eval_dataset=val_reddit_dataset,
        data_collator=data_collator,
        tokenizer=distilbart_tokenizer,
        compute_metrics=compute_metrics
    )

    # And we can run our trainer
    logging.info("Started training...")
    result = trainer.train(resume_from_checkpoint = True)
    logging.info("Done")
    logging.info(f"Time: {result.metrics['train_runtime']:.2f}")
    logging.info(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")

    logging.info(f"Pushing to hub...")
    trainer.push_to_hub()
    
    
    
    
