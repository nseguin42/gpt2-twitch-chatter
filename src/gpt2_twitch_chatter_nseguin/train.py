#!/bin/python3

from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, get_linear_schedule_with_warmup
from datasets import load_from_disk
import numpy as np
from math import ceil
import argparse
import pathlib
import os

# Parse command line arguments
parser = argparse.ArgumentParser(
    description='Train a GPT-2 model on tokenized Twitch logs created by tokenize_logs.py'
)
parser.add_argument(
    '--data_dir',
    type=pathlib.Path,
    help='Path of data directory containing logs, processed, and tokenized_datasets subdirs.',
)

parser.add_argument(
    '--datasets_dir',
    type=pathlib.Path,
    help='Path of directory containing raw logs if data_dir is unset.',
)
parser.add_argument(
    '--output_dir',
    type=pathlib.Path,
    help='Path of directory to save .csv files to if data_dir is unset.',
)
parser.add_argument(
    '--deepspeed',
    type=pathlib.Path,
    help="Path to deepspeed config. If unset, do not use deepspeed.",
)
parser.add_argument(
    '-v',
    help='Verbose mode.',
    action=argparse.BooleanOptionalAction,
)
args = parser.parse_args()


if args.data_dir:
    # make sure data_dir is a directory
    if not args.data_dir.is_dir():
        raise ValueError(f'{args.data_dir} is not a directory')
    # If data_dir is specified, use it to set datasets_dir and output_dir
    args.datasets_dir = args.data_dir / 'tokenized_datasets'
    args.output_dir = args.data_dir / 'output'

# Make sure logs and output dir are set
if (not args.datasets_dir) or (not args.output_dir):
    raise ValueError('Must specify either data_dir or both datasets_dir and output_dir.')

# Make sure logs and output dir are directories
if (not args.datasets_dir.is_dir()) or (not args.output_dir.is_dir()):
    raise ValueError('datasets_dir and output_dir must be directories.')


tokenizer = GPT2Tokenizer.from_pretrained(
    "gpt2",
    model_max_length=512,
    vocab_size=50257,
    pad_token_id=50257,
    pad_token="[PAD]"
)

model = GPT2LMHeadModel.from_pretrained(
    "gpt2-medium",
    pad_token_id=50257,
    vocab_size=50257,
    max_length=512
)

if args.deepspeed:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "9994"  # modify if RuntimeError: Address already in use
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"


model.resize_token_embeddings(len(tokenizer))
model.train()


def train_model(model, epochs=1):
    tokenized_datasets = load_from_disk(args.datasets_dir)

    batch_size = 1
    learning_rate = 1e-3

    training_args = TrainingArguments(
        seed=42,
        output_dir=args.output_dir,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        # eval_steps=steps_per_epoch//4,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        optim="adamw_torch",
        load_best_model_at_end=True,
    )

    if args.deepspeed:
        training_args.deepspeed = args.deepspeed

    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
    )
    trainer.place_model_on_device = False

    # make sure each element of our datasets is the right size
    for dataset in tokenized_datasets:
        for i in range(len(tokenized_datasets[dataset])):
            assert len(tokenized_datasets[dataset][i]['input_ids']
                       ) == 512, f'input_ids length is {len(tokenized_datasets[dataset][i]["input_ids"])}, {tokenized_datasets[dataset][i]["input_ids"]}'
            assert len(tokenized_datasets[dataset][i]['attention_mask']
                       ) == 512, f'attention_mask length is {len(tokenized_datasets[dataset][i]["attention_mask"])}'
    trainer.train()
    trainer.save_model(args.output_dir / "model")
    tokenizer.save_pretrained(args.output_dir / "model")


def main():
    train_model(model)


if __name__ == "__main__":
    main()
