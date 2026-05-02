import argparse

import torch

from common.sanskrit_texts import Datasources
from common.loss import FocalLoss
from common.model_trainer import Trainer
from common.models_factory import DEFAULT_TAGGER_MODEL_NAME, load_tagger_model
from common.pos_datasets import INDEX_PAD, PosDataloaders
from pos_taggers import bilstm_pos_tagger
from sanskrit_tagger.pos_tagger import POSTagger, get_device
from sanskrit_tagger.tagger_factory import get_pos_tagger

from auto_push import git_push_results

_train = False
_push = True

test_pos_sentences = [
    'atha kanyā pradāne sa tam eva arthaṁ vicintayan',
    'samāninye ca tat sarvaṁ bhāṇḍaṁ vaivāhikaṁ nṛpaḥ',
    'śrutvā vas tu samagraṃ tad dharma ātmā dharma saṃhitam',
    'sveditaḥ marditaḥ ca eva rañjubhiḥ pariveṣṭitaḥ',
    'ete vo gaṇitā vāsā yatra yatra nivatsyatha',
    'kiṃ kāryaṃ brūta bhavatāṃ bhayaṃ nāśayitāsmi vaḥ',
]
test_pos_sentences_tokenized = [sent.split() for sent in test_pos_sentences]

model_name=DEFAULT_TAGGER_MODEL_NAME

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='../../texts/')
    parser.add_argument("--vocab_size", type=int, default=58) 
    parser.add_argument("--labels_num", type=int, default=745) 
    parser.add_argument("--max_batches_per_epoch_train", type=int, default=100000)
    parser.add_argument("--max_batches_per_epoch_val", type=int, default=10000)
    parser.add_argument("--epoch_n", type=int, default=100)
    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--model", type=str, default='bilstm') # bilstm, cnn
    parser.add_argument("--full_pos", type=bool, default=True)
    parser.add_argument("--with_metrics", type=bool, default=True)
    parser.add_argument("--max_tokens_per_batch", type=int, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--train_tuning", type=bool, default=True)
    args = parser.parse_args()
    
    device=get_device(args.device)

    model_name = args.model_name if args.model_name else DEFAULT_TAGGER_MODEL_NAME

    if _train:

        texts = Datasources.get_datasource_list()

        max_tokens_per_batch = args.max_tokens_per_batch if args.max_tokens_per_batch else 1500

        datasets = PosDataloaders(texts, max_tokens=max_tokens_per_batch)

        if args.train_tuning:
            model = load_tagger_model(f'output/{model_name}.pth', device)
            if len(model.char2id) != datasets.vocab_size or len(model.unique_tags) != datasets.labels_num:
                raise ValueError("Размеры словарей данных и предобученной модели не совпадают!")
        else:
            model = bilstm_pos_tagger.get_model(datasets.vocab_size, datasets.labels_num, embedding_size=args.embedding_size)

        trainer = Trainer(datasets, model,
                        FocalLoss(gamma=2.5, ignore_index=INDEX_PAD), 
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

        pos_tagger = POSTagger(model, datasets.char2id, datasets.unique_tags)
    else:
        model = load_tagger_model(f'output/{model_name}.pth', device)

        pos_tagger = get_pos_tagger(model)

    for sent_tokens, sent_tags in zip(test_pos_sentences_tokenized, pos_tagger(test_pos_sentences)):
        print(' '.join('{}-{}'.format(tok, tag) for tok, tag in zip(sent_tokens, sent_tags)))
        print()
    

if __name__ == "__main__":
    _train = True
    main()
    if _push:
        git_push_results(f'output/{model_name}.pth')