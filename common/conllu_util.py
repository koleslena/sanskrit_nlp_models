from conllu import parse
import pandas as pd
import os

def get_files_conllu(target_directory):
    return [os.path.join(target_directory, item) for item in os.listdir(target_directory) if os.path.isfile(os.path.join(target_directory, item)) and item.endswith('conllu')]

def read_conllu_file(nfiles):

    nsentences = []

    for nfile in nfiles:
        # Read the content of your CoNLL-U file
        with open(nfile, "r", encoding="utf-8") as f:
            data = f.read()

        # Parse the CoNLL-U data
        nsentences.append(parse(data))

    sentences = [sent for nsent in nsentences for sent in nsent]

    # Convert to a list of dictionaries for DataFrame creation
    all_tokens = []
    for sentence in sentences:
        for token in sentence:
            all_tokens.append(token)

    # Create a Pandas DataFrame
    df = pd.DataFrame(all_tokens)
    
    return df, sentences, all_tokens

