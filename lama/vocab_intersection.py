# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from lama.modules import build_model_by_name
from tqdm import tqdm
import argparse
import spacy
import lama.modules.base_connector as base
import os


CASED_MODELS = [{
        "lm": "maskedlm",
        "label": label,
        "models_names": ["maskedlm"],
        "model_name": model_name,} for label, model_name in [
            ("roberta-base","roberta-base"), 
            ("roberta-large", "roberta-large"),
            #("longformer-base","allenai/longformer-base-4096"), 
            #("longformer-large", "allenai/longformer-large-4096"),
            ("distilroberta-base","distilroberta-base"), 
            ("bert-base-cased", "bert-base-cased"),
            ("bert-large-cased","bert-large-cased"), 
            ("distilbert-base-cased", "distilbert-base-cased"),
            #("xlnet-base-cased", "xlnet-base-cased"),
            #("xlnet-large-cased", "xlnet-large-cased"),
            ("bart-base", "facebook/bart-base"),
            ("bart-large", "facebook/bart-large"),
            ("t5-small","t5-small"),
            ("t5-base","t5-base"),
            ("t5-large","t5-large"),
    ]
] + [
      {
          "lm": "causallm",
          "label": label,
          "models_names": ["causallm"],
          "model_name": model_name} for label, model_name in [
              ("gpt2", "gpt2"),
              ("gpt2-medium","gpt2-medium"),
              ("gpt2-large","gpt2-large"),
          ]
  ] + [
      {
          "lm": "causallm",
          "label": "gpt2-xl",
          "models_names": ["causallm"],
          "model_name": "gpt2-xl",
          "batch_size": 32
          }
]
 
elmo = [
 {
    # "ELMO ORIGINAL"
    "lm": "elmo",
    "elmo_model_dir": "pre-trained_language_models/elmo/original",
    "elmo_model_name": "elmo_2x4096_512_2048cnn_2xhighway",
    "elmo_vocab_name": "vocab-2016-09-10.txt",
    "elmo_warm_up_cycles": 5
  },
  {
    # "ELMO ORIGINAL 5.5B"
    "lm": "elmo",
    "elmo_model_dir": "pre-trained_language_models/elmo/original5.5B/",
    "elmo_model_name": "elmo_2x4096_512_2048cnn_2xhighway_5.5B",
    "elmo_vocab_name": "vocab-enwiki-news-500000.txt",
    "elmo_warm_up_cycles": 5
  }
]

CASED_COMMON_VOCAB_FILENAME = "pre-trained_language_models/common_vocab_cased_SECOND.txt"

LOWERCASED_MODELS = [
    {
        "lm": "maskedlm",
        "label": "google/multiberts-seed_0",
        "models_names": ["maskedlm"],
        "model_name": "google/multiberts-seed_0",
        }
]

LOWERCASED_COMMON_VOCAB_FILENAME = "pre-trained_language_models/common_vocab_lowercased.txt"


def __vocab_intersection(models, filename):

    vocabularies = []

    for arg_dict in models:

        args = argparse.Namespace(**arg_dict)
        print(args)
        model = build_model_by_name(args.lm, args)

        vocabularies.append(model.vocab)
        print(type(model.vocab))

    if len(vocabularies) > 0:
        common_vocab = set(vocabularies[0])
        for vocab in vocabularies:
            common_vocab = common_vocab.intersection(set(vocab))

        # no special symbols in common_vocab
        for symbol in base.SPECIAL_SYMBOLS:
            if symbol in common_vocab:
                common_vocab.remove(symbol)

        # remove stop words
        from spacy.lang.en.stop_words import STOP_WORDS
        for stop_word in STOP_WORDS:
            if stop_word in common_vocab:
                print(stop_word)
                common_vocab.remove(stop_word)

        common_vocab = list(common_vocab)

        # remove punctuation and symbols
        nlp = spacy.load('en')
        manual_punctuation = ['(', ')', '.', ',']
        new_common_vocab = []
        for i in tqdm(range(len(common_vocab))):
            word = common_vocab[i]
            doc = nlp(word)
            token = doc[0]
            if(len(doc) != 1):
                print(word)
                for idx, tok in enumerate(doc):
                    print("{} - {}".format(idx, tok))
            elif word in manual_punctuation:
                pass
            elif token.pos_ == "PUNCT":
                print("PUNCT: {}".format(word))
            elif token.pos_ == "SYM":
                print("SYM: {}".format(word))
            else:
                new_common_vocab.append(word)
            # print("{} - {}".format(word, token.pos_))
        common_vocab = new_common_vocab

        # store common_vocab on file
        with open(filename, 'w') as f:
            for item in sorted(common_vocab):
                f.write("{}\n".format(item))


def main():
    # cased version
    __vocab_intersection(CASED_MODELS, CASED_COMMON_VOCAB_FILENAME)
    # lowercased version
    #__vocab_intersection(LOWERCASED_MODELS, LOWERCASED_COMMON_VOCAB_FILENAME)


if __name__ == '__main__':
    main()
