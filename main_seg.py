import argparse

from common.models_factory import load_segmenter_model, DEFAULT_SEGMENTER_MODEL_NAME
from common.segmenter_datasets import SegmenterDataloaders

from common.sanskrit_texts import Datasources
from sanskrit_tagger.pos_tagger import get_device
from sanskrit_tagger.tagger_factory import Segmenter, get_segmenter

from common.segmenter_trainer import SegmenterTrainer
from segmenter.segmenter import SanskritPointerSegmenter

from auto_push import git_push_results

_train = False
_push = True

model_name = DEFAULT_SEGMENTER_MODEL_NAME

test_splitter_sentences = [
    "tamuktavantamātreyamagniveśa uvāca ha",
    "bhagavan laṅghanaṁ kiṁsvillaṅghanīyāśca kīdṛśāḥ",
    "bṛṁhaṇaṁ bṛṁhaṇīyāśca rūkṣaṇīyāśca rūkṣaṇam",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_batches_per_epoch_train", type=int, default=100000)
    parser.add_argument("--max_batches_per_epoch_val", type=int, default=10000)
    parser.add_argument("--epoch_n", type=int, default=100)
    parser.add_argument("--train_tuning", type=bool, default=False)
    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--max_tokens_per_batch", type=int, default=None)
    parser.add_argument("--device", type=str, default='mps')
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--version", type=str, default=None)
    args = parser.parse_args()

    device=get_device(args.device)

    model_name = args.model_name if args.model_name else DEFAULT_SEGMENTER_MODEL_NAME

    if _train:
            
        texts = Datasources.get_datasource_list() 

        max_tokens_per_batch = args.max_tokens_per_batch if args.max_tokens_per_batch else 1000

        datasets = SegmenterDataloaders(texts, max_tokens_per_batch=max_tokens_per_batch)

        if args.train_tuning:
            model = load_segmenter_model(f'segmenter_output/{model_name}.pth', device)
            if len(model.char2id) + 1 != datasets.vocab_size:
                raise ValueError("Размеры словарей данных и предобученной модели не совпадают!")
        else:
            model = SanskritPointerSegmenter(datasets.get_vocab_size(), args.embedding_size, device).to(device)
            
        trainer = SegmenterTrainer( datasets, 
                                    model,
                                    output_model_name=model_name, 
                                    device=device,
                                    lr=5e-5,
                                    epoch_n=args.epoch_n,
                                    early_stopping_patience=5,
                                    max_batches_per_epoch_train=args.max_batches_per_epoch_train,
                                    max_batches_per_epoch_val=args.max_batches_per_epoch_val)

        trainer.train()

        segmenter = Segmenter(model, datasets.get_char2id())
    else:
        model = load_segmenter_model(f'segmenter_output/{model_name}.pth', device)

        segmenter = get_segmenter(model)

    for sent, predicted in zip(test_splitter_sentences, segmenter(test_splitter_sentences)):
        print(f"Input:  {sent}")
        print(f"Output: {predicted}")

if __name__ == "__main__":
    _train = True
    main()
    if _push:
        git_push_results(f'output/{model_name}.pth')

