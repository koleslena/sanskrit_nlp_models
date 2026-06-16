from conllu import parse
import pandas as pd
import os
from itertools import product

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

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

    sentences = [sent for nsent in nsentences for sent in nsent if 'text' in sent.metadata]
    sents = [transliterate(sent.metadata['text'], sanscript.IAST, sanscript.SLP1) for sent in sentences]

    # Convert to a list of dictionaries for DataFrame creation
    all_tokens = []
    for sentence in sentences:
        with_tasil = False
        text = transliterate(sentence.metadata['text'], sanscript.IAST, sanscript.SLP1)
        if 'taH ' in text or 'taSca' in text or 'tas' in text or 'to ' in text:
            with_tasil = True
        for token in sentence:
            token['sent_id'] = sentence.metadata['sent_id']
            form = token['form'] if len(token['form']) > 0 and token['form'] != '_' else token['lemma']
            token['form_slp1'] = transliterate(form, sanscript.IAST, sanscript.SLP1)
            if with_tasil and token['form_slp1'].endswith('At'):
                abls = set([token['form_slp1'][:-2] + 'atas', token['form_slp1'][:-2] + 'ato', token['form_slp1'][:-2] + 'ataS', token['form_slp1'][:-2] + 'ataH'])
                found = [a for a in abls if a in text]
                if len(found) > 0:
                    token['form_slp1'] = token['form_slp1'][:-2] + 'ataH'
            all_tokens.append(token)

    # Create a Pandas DataFrame
    df = pd.DataFrame(all_tokens)
    
    return df, sents

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

def token_variants(token):
    # 1. Выделяем основу (убираем первый символ и два последних)
    # Для 'gacCati' -> 'acCat'
    core = token[1:-2]

    # 2. Определяем префиксы. 
    # Для первой группы префиксом служит первый оригинальный символ токена
    first_char = token[0] 
    prefixes = [first_char, 'A', 'o', 'e', 'C', 'U', 'I']

    # 3. Определяем суффиксы флексий (окончаний)
    suffixes = ['ata ', 'atas', 'ato', 'ataS', 'ataz', 'ataH']

    # 4. Генерируем весь массив в один проход
    variants = [
        f"{pref}{core}{suff}" 
        for pref, suff in product(prefixes, suffixes)
    ]

    return variants

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
        with_tasil = False
        text = IASTToSlp(sentence.metadata['text'])
        if 'taH ' in text or 'taSca' in text or 'tas' in text or 'to ' in text:
            with_tasil = True
        all_tokens = []
        for token in sentence:
            if str(token['id']).isdigit():
                form  = token['form'] if len(token['form']) > 0 and token['form'] != '_' else token['lemma']
                token_form = IASTToSlp(form) if transliterate else form
                if with_tasil and token_form.endswith('At') and token['upos'] in ['NOUN', 'ADJ', 'NUM', 'PRON']:
                    abls = set(token_variants(token_form))
                    found = [a for a in abls if a in text]
                    if len(found) > 0:
                        token_form = token_form[:-2] + 'ataH'
                token_form = token_form.replace("'", 'a', 1)
                all_tokens.append(token_form)
                all_tokens.append("-" if token['feats'] == {'Case': 'Cpd'} else " ")
        sent = ''.join(all_tokens[:-1])
        
        if sentence.metadata['sent_id'] == '62444':
            sent = sent.replace('SaktyAH', 'SaktitaH').replace('vayasaH', 'vayastaH')
        if sentence.metadata['sent_id'] == '298642':
            sent = sent.replace('nitya-tvataH', 'nitya-tvAt')
        if sentence.metadata['sent_id'] == '536902':
            sent = sent.replace('svAtantrya-sAra-tvataH', 'svAtantrya-sAra-tvAt')
        
        splited.append(sent)

    data = {
        'sent_id': ids,
        'src': sents,
        'trg': splited
    }
    df = pd.DataFrame(data)
    return clean_df(df)

