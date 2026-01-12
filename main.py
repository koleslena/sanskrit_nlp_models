import argparse

import torch
from torch.nn import functional as F

from common.datasets_util import get_pos_datasets, get_split_pos_datasets, Datasources
from common.model_trainer import Trainer
from pos_taggers import cnn_pos_tagger
from pos_taggers import bilstm_pos_tagger
from sanskrit_tagger.pos_tagger import POSTagger
from sanskrit_tagger.tagger_factory import get_pos_tagger

_train = False

test_pos_sentences = [
    'atha kanyā pradāne sa tam eva arthaṁ vicintayan',
    'samāninye ca tat sarvaṁ bhāṇḍaṁ vaivāhikaṁ nṛpaḥ',
    'śrutvā vas tu samagraṃ tad dharma ātmā dharma saṃhitam',
    'sveditaḥ marditaḥ ca eva rañjubhiḥ pariveṣṭitaḥ',
    'ete vo gaṇitā vāsā yatra yatra nivatsyatha',
    'kiṃ kāryaṃ brūta bhavatāṃ bhayaṃ nāśayitāsmi vaḥ',
]
test_pos_sentences_tokenized = [sent.split() for sent in test_pos_sentences]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=42) 
    parser.add_argument("--labels_num", type=int, default=734) 
    parser.add_argument("--max_batches_per_epoch_train", type=int, default=400)
    parser.add_argument("--max_batches_per_epoch_val", type=int, default=100)
    parser.add_argument("--epoch_n", type=int, default=50)
    parser.add_argument("--embedding_size", type=int, default=64)
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--model", type=str, default='bilstm') # bilstm, cnn
    parser.add_argument("--full_pos", type=bool, default=True)
    parser.add_argument("--with_metrics", type=bool, default=True)
    args = parser.parse_args()

    model_name = cnn_pos_tagger.get_model_name(full_pos=args.full_pos) if args.model == 'cnn' else bilstm_pos_tagger.get_model_name(full_pos=args.full_pos)

    if _train:
        datasets = get_split_pos_datasets([
                                            Datasources.MAHABHARATA, 
                                           Datasources.RAMAYANA, 
                                           Datasources.AMARAKOSHA, 
                                           Datasources.HITOPADESHA, 
                                           Datasources.SHIVAPURANA, 
                                           Datasources.BHAGAVATAPURANA, 
                                           Datasources.VISHNUPURANA
                                           ], 
                            full_pos=args.full_pos)
        
        if args.model == 'cnn':
            model = cnn_pos_tagger.get_model(datasets.vocab_size, datasets.labels_num, embedding_size=args.embedding_size)
        else:
            model = bilstm_pos_tagger.get_model(datasets.vocab_size, datasets.labels_num, embedding_size=args.embedding_size)

        trainer = Trainer(datasets, model,
                        F.cross_entropy, 
                        output_model_name=model_name, 
                        device=args.device,
                        with_metrics=args.with_metrics,
                        lr=5e-4,
                        epoch_n=args.epoch_n,
                        early_stopping_patience=5,
                        max_batches_per_epoch_train=args.max_batches_per_epoch_train,
                        max_batches_per_epoch_val=args.max_batches_per_epoch_val,
                        lr_scheduler_ctor=lambda optim: torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=2, factor=0.5))

        trainer.train()

        model.load_state_dict(torch.load(f'output/{model_name}.pth'))

        pos_tagger = POSTagger(model, datasets.char2id, datasets.unique_tags, datasets.max_sent_len, datasets.max_origin_token_len)
    else:
        model = cnn_pos_tagger.get_model(args.vocab_size, args.labels_num, embedding_size=args.embedding_size) if args.model == 'cnn' else \
            bilstm_pos_tagger.get_model(args.vocab_size, args.labels_num, embedding_size=args.embedding_size)

        model.load_state_dict(torch.load(f'output/{model_name}.pth'))

        pos_tagger = get_pos_tagger(model, args.vocab_size, args.labels_num)

    for sent_tokens, sent_tags in zip(test_pos_sentences_tokenized, pos_tagger(test_pos_sentences)):
        print(' '.join('{}-{}'.format(tok, tag) for tok, tag in zip(sent_tokens, sent_tags)))
        print()


if __name__ == "__main__":
    _train = True
    main()