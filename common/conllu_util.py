from conllu import parse
import pandas as pd
import os

from common.transliterate import IASTToSlp

def get_files_conllu(target_directory):
    return [os.path.join(target_directory, item) for item in os.listdir(target_directory) if os.path.isfile(os.path.join(target_directory, item)) and item.endswith('conllu')]

def read_pos_conllu_file(nfiles):

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

def clean_df(df):
    # 1. Считаем длины
    src_len = df['src'].str.split().str.len()
    trg_len = df['trg'].str.split().str.len()

    # 2. Убираем пустые или слишком короткие (меньше 2 слов)
    bad_empty = (src_len < 2) | (trg_len < 2)

    # 3. Убираем аномальную разницу 
    bad_ratio = (src_len > trg_len)

    # 4. Собираем все ID «плохих» строк
    exclude_sent_id = df[bad_empty | bad_ratio]['sent_id']

    # 5. Оставляем только хорошие данные
    df_clean = df[~df['sent_id'].isin(exclude_sent_id)]

    print(f"Удалено строк: {len(exclude_sent_id)} из {len(df)}")
    return df_clean

def read_split_conllu_file(nfiles, transliterate=False):
    nsentences = []

    for nfile in nfiles:
        # Read the content of your CoNLL-U file
        with open(nfile, "r", encoding="utf-8") as f:
            data = f.read()

        # Parse the CoNLL-U data
        nsentences.append(parse(data))
    
    sentences = [sent for nsent in nsentences for sent in nsent if 'text' in sent.metadata]
    sents = [IASTToSlp(sent.metadata['text']) if transliterate else sent.metadata['text'] for sent in sentences]
    ids = [sent.metadata['sent_id'] for sent in sentences]
    splited = []
    for sentence in sentences:
        all_tokens = []
        for token in sentence:
            if str(token['id']).isdigit():
                form  = token['form'] if len(token['form']) > 0 and token['form'] != '_' else token['lemma']
                token_form = IASTToSlp(form) if transliterate else form
                token_form = token_form.replace("'", 'a', 1)
                all_tokens.append(token_form)
                all_tokens.append("-" if token['feats'] == {'Case': 'Cpd'} else " ")
        sent = ''.join(all_tokens[:-1])
        splited.append(sent)

    data = {
        'sent_id': ids,
        'src': sents,
        'trg': splited
    }
    df = pd.DataFrame(data)
    return clean_df(df)

