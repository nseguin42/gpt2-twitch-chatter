#!/bin/python3

import argparse
import pathlib
import os
import re
import pandas as pd
import multiprocessing as mp

# Parse command line arguments
parser = argparse.ArgumentParser(
    description='Parse raw twitch logs and save them as a .csv.'
)
parser.add_argument(
    '--data_dir',
    type=pathlib.Path,
    help='Path of data directory containing logs, processed, and tokenized_datasets subdirs.',
)

parser.add_argument(
    '--logs_dir',
    type=pathlib.Path,
    help='Path of directory containing raw logs if data_dir is unset.',
)
parser.add_argument(
    '--output_dir',
    type=pathlib.Path,
    help='Path of directory to save .csv files to if data_dir is unset.',
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


if args.data_dir:
    # make sure data_dir is a directory
    if not args.data_dir.is_dir():
        raise ValueError(f'{args.data_dir} is not a directory')
    # If data_dir is specified, use it to set logs_dir and output_dir
    args.logs_dir = args.data_dir / 'logs'
    args.output_dir = args.data_dir / 'processed'

# Make sure logs and output dir are set
if (not args.logs_dir) or (not args.output_dir):
    raise ValueError('Must specify either data_dir or both logs_dir and output_dir.')

# Make sure logs and output dir are directories
if (not args.logs_dir.is_dir()) or (not args.output_dir.is_dir()):
    raise ValueError('logs_dir and output_dir must be directories.')


# Count the number of files in logs_dir
def count_logs():
    num_logs = len(os.listdir(args.logs_dir))
    print(f'Found {num_logs} logs in {args.logs_dir}')


# use a Python generator to read the file line by line
def read_lines(file):
    with open(file, "r") as f:
        for line in f:
            yield line


# each message looks like:
# [2021-01-1 00:00:00] #channel username: message
# we want to capture the (date, time, channel, username, message)
msg_pattern = r"\[(\d{4}-\d{2}-\d{1,2}) (\d{2}:\d{2}:\d{2})\] #(\w+) (\w+): (.*)"
msg_prog = re.compile(msg_pattern)
# similarly capture ban information
ban_pattern = r"\[(\d{4}-\d{2}-\d{1,2}) (\d{2}:\d{2}:\d{2})\] #(\w+) (\w+) has been (timed out for \d+ .*|banned)"
ban_prog = re.compile(ban_pattern)


def recent_messages(row, df):
    return df[(df["channel"] == row["channel"]) & (df["date"] == row["date"]) & (df["time"] < row["time"])].tail(100)


def handle_ban(row, df):
    if "minutes" in row["ban_type"] or "seconds" in row["ban_type"]:
        # get index of user messages within 100 messages of the ban, then drop them
        recent = recent_messages(row, df).where(df["username"] == row["username"]).dropna()
        df = df.drop(recent.index)
    else:
        # drop user messages from the day of the ban
        df = df.mask((df["username"] == row["username"]) & (
            df["date"] == row["date"]) & (df["time"] < row["time"])).dropna()


def parse_msgs(file):
    df = pd.DataFrame(columns=["date", "time", "channel", "username", "message"])
    for line in read_lines(file):
        msg_match = msg_prog.match(line)
        if msg_match:
            groups = msg_match.groups()
            yield groups


def parse_bans(file):
    bans = pd.DataFrame(columns=["date", "time", "channel", "username", "ban_type"])
    for line in read_lines(file):
        ban_match = ban_prog.match(line)
        if ban_match:
            groups = ban_match.groups()
            yield groups


def parse_file(file):
    df = pd.DataFrame(parse_msgs(file), columns=["date", "time", "channel", "username", "message"])
    bans = pd.DataFrame(parse_bans(file), columns=[
                        "date", "time", "channel", "username", "ban_type"])
    for index, row in bans.iterrows():
        handle_ban(row, df)

    print("Finished parsing file:", file.name)
    return df, file


def save_result(result):
    df, file = result
    out_path = args.output_dir / (file.stem + ".csv")
    if not args.d:
        df.to_csv(out_path, index=False)
        if args.v:
            print("Saving file: ", out_path)
    else:
        # print out the df
        print(df)


def __main__():
    if args.v:
        count_logs()

    if args.n:
        pool = mp.Pool(processes=args.n)
    else:
        pool = mp.Pool()

    # iterate over input_dirs in parallel
    results = []
    for log in os.listdir(args.logs_dir):
        log_path = args.logs_dir / log
        # r = pool.apply_async(parse_file, args=(log_path,),
        #                     callback=save_result, error_callback=print)
        pool.apply(parse_file, args=(log_path,))
        # results.append(r)

    pool.close()
    pool.join()


if __name__ == '__main__':
    __main__()
