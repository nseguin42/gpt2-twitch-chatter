#!/bin/python3
from transformers import GPT2Tokenizer
import pandas as pd
import numpy as np
import multiprocessing as mp
import argparse
import pathlib
import os
from datasets import Dataset, concatenate_datasets

# Parse command line arguments
parser = argparse.ArgumentParser(
    description='Tokenize parsed Twitch logs created with parse_logs.py and save a DatasetDict.'
)

parser.add_argument(
    '--data_dir',
    type=pathlib.Path,
    help='Path of data directory containing logs, processed, and tokenized_datasets subdirs.',
)

parser.add_argument(
    '--logs_dir',
    type=pathlib.Path,
    help='Path of directory containing processed logs if data_dir is unset.',
)
parser.add_argument(
    '--output_dir',
    type=pathlib.Path,
    help='Path of directory to save tokenized datasets to if data_dir is unset.',
)
parser.add_argument(
    '-v',
    help='Verbose mode.',
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    '-d',
    help='Dry run. Do not save any files.',
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    '-n',
    help='Number of multiprocessing workers to use. If unset, use all available threads.',
    action='store',
    type=int,
)
args = parser.parse_args()

if args.n:
    numprocs = args.n
else:
    numprocs = mp.cpu_count()

if args.data_dir:
    # make sure data_dir is a directory
    if not args.data_dir.is_dir():
        raise ValueError(f'{args.data_dir} is not a directory')
    # If data_dir is specified, use it to set logs_dir and output_dir
    args.logs_dir = args.data_dir / 'processed'
    args.output_dir = args.data_dir / 'tokenized_datasets'

# Make sure logs and output dir are set
if (not args.logs_dir) or (not args.output_dir):
    raise ValueError('Must specify either data_dir or both logs_dir and output_dir.')

# Make sure logs and output dir are directories
if (not args.logs_dir.is_dir()) or (not args.output_dir.is_dir()):
    raise ValueError('logs_dir and output_dir must be directories.')


# IMPORTANT: Make sure we resize our model embeddings to match the tokenizer's max length later!
tokenizer = GPT2Tokenizer.from_pretrained(
    "gpt2",
    model_max_length=512,
    vocab_size=50257,
    pad_token_id=50257,
    pad_token="[PAD]"
)

vocab_size = tokenizer.vocab_size
block_size = tokenizer.model_max_length
newline_token = tokenizer.encode("\n")[0]


# disable tokenizer multiprocessing / suppress warning
# we're going to do parallel single-threaded tokenization on each core later.
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# join two nparrays with a token in between
def splice_arrays(ar1, ar2, sep):
    return np.concatenate((ar1, np.array([sep]), ar2))


# begin a new chunk with just a BOS token
def make_fresh_chunk():
    return pd.Series({"input_ids": np.array([tokenizer.bos_token_id], dtype=np.int32), "attention_mask": np.array([1], dtype=np.int32)})


# appends row to chunk and returns the new chunk
def add_row_to_chunk(chunk, row):
    input_ids = splice_arrays(chunk.input_ids, row.input_ids, newline_token)
    attention_mask = splice_arrays(chunk.attention_mask, row.attention_mask, 1)
    return pd.Series([input_ids, attention_mask], index=["input_ids", "attention_mask"])


# appends an EOS token with attention mask 1 to the end of the chunk
def add_eos_token(chunk):
    input_ids = splice_arrays(chunk.input_ids, np.array([tokenizer.eos_token_id]), newline_token)
    attention_mask = splice_arrays(chunk.attention_mask, np.array([1]), 1)
    return pd.Series([input_ids, attention_mask], index=["input_ids", "attention_mask"])


# add pad tokens to the end of the chunk until it's 512 tokens long
def pad_chunk(chunk, num_pad_tokens):
    input_ids = np.pad(chunk.input_ids, (0, num_pad_tokens),
                       constant_values=tokenizer.pad_token_id)
    attention_mask = np.pad(chunk.attention_mask, (0, num_pad_tokens), constant_values=-100)
    padded_chunk = pd.Series([input_ids, attention_mask], index=["input_ids", "attention_mask"])

    return padded_chunk


# make sure the input_ids and attention_mask are at most 512 tokens long
# note: pass the row as a pd.Series, not a pd.DataFrame.
def truncate_chunk(chunk):
    # assert type(chunk) == pd.Series

    # remove an extra char and replace it with EOS token
    input_ids = chunk["input_ids"][:block_size - 1]
    input_ids = np.concatenate([input_ids, [tokenizer.eos_token_id]])
    attention_mask = chunk["attention_mask"][:block_size]

    return pd.Series([input_ids, attention_mask], index=["input_ids", "attention_mask"])


# pad or truncate the chunk to the block size
# and make sure it ends with the EOS token
def finish_chunk(chunk):
    # chunk["inupt_ids"] might be passed as a list or nparray
    # so we need to handle both cases
    chunk = add_eos_token(chunk)
    try:
        chunk_length = chunk["input_ids"].shape[0]
    except:
        chunk_length = len(chunk["input_ids"])

    #padded = False
    #truncated = False

    if chunk_length < block_size:
        chunk = pad_chunk(chunk, block_size - chunk_length)
        #padded = True
    else:
        chunk = truncate_chunk(chunk)
        #truncated = True

    # make sure our chunks are all the right length
    # assert len(chunk["input_ids"]
    #           ) == block_size, f'chunk length is {len(chunk["input_ids"]), chunk["input_ids"]}, {padded}, {truncated}'
    return chunk.transpose()


# yield each df and the filename it came from
def iterate_dfs(logs_dir):
    for log_file in logs_dir.glob('*.csv'):
        if args.v:
            print("Tokenizing {}".format(log_file.name))
        df = pd.read_csv(log_file)
        yield df, log_file


# yield a single line of chat
def iterate_rows(df):
    for row in df.iterrows():
        yield row


# yield a single input sequence for the model
def chunk_generator(df):
    chunk = make_fresh_chunk()
    for idx, row in df.iterrows():
        # truncate individual lines, leaving 3 tokens to make room for newline and BOS/EOS tokens
        tokenized = tokenizer(
            row["message"],
            truncation=True,
            max_length=block_size-3,
            add_special_tokens=False
        )
        # make sure to leave room for the final newline and eos token
        if len(chunk.input_ids) + len(tokenized.input_ids) >= block_size - 1:
            # we can't add the row, so finish the chunk and yield it
            yield finish_chunk(chunk)
            # start the next chunk with a BOS token
            chunk = make_fresh_chunk()
        # add the row to the chunk
        chunk = add_row_to_chunk(chunk, tokenized)


def tokenize_df(df, file):
    return pd.DataFrame(chunk_generator(df)), file


def prepare_tokenized_data():
    result_list = []

    def log_result(result):
        df, file = result
        if args.v:
            print("Finished processing {} ({} chunks)".format(file.name, len(df)))
        # convert to Dataset
        dataset = Dataset.from_pandas(df, split="train")
        result_list.append(dataset)

    pool = mp.Pool(processes=numprocs)

    for df, log_file in iterate_dfs(args.logs_dir):
        pool.apply_async(tokenize_df, args=(df, log_file,),
                         callback=log_result, error_callback=print)
    pool.close()

    # wait for all processes to finish
    pool.join()

    return concatenate_datasets(result_list)


def __main__():
    num_files = len(list(args.logs_dir.glob('*.csv')))
    print("Tokenizing {} files with {} processes.".format(num_files, numprocs))
    tokenized_dataset = prepare_tokenized_data()
    tokenized_datasets = tokenized_dataset.train_test_split(test_size=0.1)

    if not args.d:
        tokenized_datasets.save_to_disk(args.output_dir)


if __name__ == "__main__":
    __main__()
