import argparse
import pathlib
import pandas as pd
import os
import re
import multiprocessing as mp

# Parse command line arguments
parser = argparse.ArgumentParser(
    description='Clean .csv files by dropping undesirable messages.'
)
parser.add_argument(
    '--data_dir',
    type=pathlib.Path,
    help='Path of data directory containing logs, processed, and tokenized_datasets subdirs.',
)

parser.add_argument(
    '--input_dir',
    type=pathlib.Path,
    help='Path of directory containing input .csv files if data_dir is unset.',
)
parser.add_argument(
    '--output_dir',
    type=pathlib.Path,
    help='Path of directory to save cleaned.csv files to if data_dir is unset.',
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
    # If data_dir is specified, use it to set input_dir
    args.input_dir = args.data_dir / 'processed'

# Make sure input dir is set
if (not args.input_dir):
    raise ValueError('Must specify either data_dir or input_dir.')

# If output dir is not set, just overwrite the input files
if (not args.output_dir):
    args.output_dir = args.input_dir

# Make sure input and output dirs are directories
if (not args.input_dir.is_dir()) or (not args.output_dir.is_dir()):
    raise ValueError('input_dir and output_dir must be directories.')


# Count the number of files in input_dir
def count_logs():
    num_logs = len(os.listdir(args.input_dir))
    print(f'Found {num_logs} logs in {args.input_dir}')
    return num_logs


def drop_empty_messages(df):
    # messages might be NaN
    df.dropna(subset=['message'], inplace=True)


# Drop all messages by bots and other blacklisted users
blacklisted_users = [
    "nightbot",
    "streamelements",
    "moobot",
    "markov_chain_bot",
    "scootycoolguy",
    "autoraver",
    "sumbot_",
    "scripbozo",
    "piebot_3000",
    "poopthefirst",
    "moonmoon_nam",
    "autoraver"
]


# matches links, from user Daveo at https://stackoverflow.com/questions/3809401/
link_pattern = r"[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)"
link_prog = re.compile(link_pattern)


def string_contains_url(string):
    print(string)
    return link_prog.search(string)


# Drop messages from users who only send a few messages (often spam/bots/chat hoppers)
def drop_sparse_users(df):
    sparse_users = df.groupby("username").count().where(
        df.groupby("username").count()["message"] < 3).dropna().index
    return df.drop(df[df.username.isin(sparse_users)].index, inplace=False)


def message_is_empty(row):
    return row['message'] == ''


def user_is_blacklisted(row):
    return row['username'] in blacklisted_users


def message_has_link(row):
    # true if there are any matches
    return link_prog.search(row['message']) is not None


def message_too_short(row):
    return len(row['message']) < 2


def message_too_long(row):
    return len(row['message']) > 300


def message_is_caps_spam(row):
    return len(row['message']) > 100 and row['message'].isupper()


def message_is_command(row):
    return row['message'].startswith('!')


def message_has_mention(row):
    return '@' in row['message']


def should_drop_line(row):
    # Return true if message satisfies any of the drop conditions
    return (
        message_is_empty(row) or
        user_is_blacklisted(row) or
        message_has_link(row) or
        # message_too_short(row) or
        # message_too_long(row) or
        # message_is_caps_spam(row) or
        message_is_command(row)  # or
        # message_has_mention(row)
    )


def drop_lines(df):
    return df[~df.apply(should_drop_line, axis=1)]


def clean_df(df):
    # drop sparse users
    clean = drop_sparse_users(df)
    # drop other lines
    clean = drop_lines(df)
    return clean


def save_df(df, path):
    df.to_csv(path, index=False)


def clean_file(in_path, out_path):
    df = pd.read_csv(in_path).astype(
        {'message': 'string', 'username': 'string'}).dropna().reset_index(drop=True)
    orig_len = len(df)
    clean = clean_df(df)
    clean_len = len(clean)
    return clean, out_path, orig_len, clean_len


def save_result(result):
    clean, out_path, orig_len, clean_len = result
    if not args.d:
        save_df(clean, out_path)
    if args.v:
        print("Finished cleaning {}. Removed {}/{} lines ({:.2f}%)".format(
            out_path, orig_len - clean_len, orig_len, 100 * (orig_len - clean_len) / orig_len))


def __main__():
    if args.v:
        num_logs = count_logs()

    if args.n:
        pool = mp.Pool(processes=args.n)
    else:
        pool = mp.Pool()

    # iterate over input_dirs in parallel
    results = []
    for i, log in enumerate(os.listdir(args.input_dir)):
        in_path = args.input_dir / log
        out_path = args.output_dir / log
        r = pool.apply_async(clean_file, args=(in_path, out_path,),
                             callback=save_result, error_callback=print)
        results.append(r)

    pool.close()
    pool.join()

    print(f'Cleaned {len(results)} logs.')


if __name__ == "__main__":
    __main__()
