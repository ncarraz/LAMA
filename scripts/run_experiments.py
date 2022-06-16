# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
from batch_eval_KB_completion import main as run_evaluation
from batch_eval_KB_completion import load_file
from lama.modules import build_model_by_name
import pprint
import statistics
from os import listdir
import os
from os.path import isfile, join
from shutil import copyfile
from collections import defaultdict


my_parser = argparse.ArgumentParser(description='Evaluate LMs on a relation')
my_parser.add_argument("--relations", help="relations file", type=str, default="data/relations.jsonl")
my_parser.add_argument("--batch-size", help="batch size", type=int, default=128)
my_parser.add_argument("--lowercase", help="lowercase samples", action="store_true")
my_parser.add_argument("--output-dir", help="output directory", type=str, default="output")
my_parser.add_argument("--data-dir", help="dataset directory", type=str, default="TREX")
my_args = my_parser.parse_args()

LMs2 = [
    {
        "lm": "causallm",
        "label": "transfo-xl-wt103",
        "models_names": ["causallm"],
        "model_name": "transfo-xl-wt103"
        } 
]

LMs = [
        {
        "lm": "maskedlm",
        "label": label,
        "models_names": ["maskedlm"],
        "model_name": model_name} for label, model_name in [
            ("roberta-base","roberta-base"), 
            ("roberta-large", "roberta-large"),
            ("longformer-base","allenai/longformer-base-4096"), 
            ("longformer-large", "allenai/longformer-large-4096"),
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
            ("gpt2","gpt2"),
            ("transfo-xl-wt103", "transfo-xl-wt103")
    ]
] 

ELMO = [
    {
        "lm": "elmo",
        "label": "elmo",
        "models_names": ["elmo"],
        "elmo_model_name": "elmo_2x4096_512_2048cnn_2xhighway",
        "elmo_vocab_name": "vocab-2016-09-10.txt",
        "elmo_model_dir": "pre-trained_language_models/elmo/original",
        "elmo_warm_up_cycles": 10,
    },
    {
        "lm": "elmo",
        "label": "elmo5B",
        "models_names": ["elmo"],
        "elmo_model_name": "elmo_2x4096_512_2048cnn_2xhighway_5.5B",
        "elmo_vocab_name": "vocab-enwiki-news-500000.txt",
        "elmo_model_dir": "pre-trained_language_models/elmo/original5.5B/",
        "elmo_warm_up_cycles": 10,
    },
]


def run_experiments(
    relations,
    data_path_pre,
    data_path_post,
    input_param={
        "lm": "bert",
        "label": "bert_large",
        "models_names": ["bert"],
        "bert_model_name": "bert-large-cased",
        "bert_model_dir": "pre-trained_language_models/bert/cased_L-24_H-1024_A-16",
    },
    use_negated_probes=False,
):
    model = None
    pp = pprint.PrettyPrinter(width=41, compact=True)

    all_Precision1 = []
    all_Precision10 = []
    type_Precision1 = defaultdict(list)
    type_count = defaultdict(list)

    output_dir = os.path.join(my_args.output_dir, "results",os.path.basename(my_args.relations).split(".")[0], input_param["label"])
    output_file = open("results/results_{}.csv".format(os.path.basename(my_args.relations)), "a")

    for relation in relations:
        pp.pprint(relation)
        PARAMETERS = {
            "dataset_filename": "{}{}{}".format(
                data_path_pre, relation["relation"], data_path_post
            ),
            "common_vocab_filename": "pre-trained_language_models/common_vocab_lowercased.txt" if my_args.lowercase else "pre-trained_language_models/common_vocab_cased.txt",
            "template": "",
            "batch_size": my_args.batch_size,
            "logdir": "output",
            "full_logdir": os.path.join(output_dir, relation["relation"]),
            "lowercase": my_args.lowercase,
            "max_sentence_length": 100,
            "threads": -1,
            "interactive": False,
            "use_negated_probes": use_negated_probes,
        }

        if "template" in relation:
            PARAMETERS["template"] = relation["template"]
            if use_negated_probes:
                PARAMETERS["template_negated"] = relation["template_negated"]

        PARAMETERS.update(input_param)
        print(PARAMETERS)

        args = argparse.Namespace(**PARAMETERS)

        # see if file exists
        try:
            data = load_file(args.dataset_filename)
        except Exception as e:
            print("Relation {} excluded.".format(relation["relation"]))
            print("Exception: {}".format(e))
            continue

        if model is None:
            [model_type_name] = args.models_names
            model = build_model_by_name(model_type_name, args)

        Precision1, Precision10 = run_evaluation(args, shuffle_data=False, model=model)
        print("P@1 : {}".format(Precision1), flush=True)
        all_Precision1.append(Precision1)
        all_Precision10.append(Precision10)

        results_file = open(os.path.join(output_dir, "result.csv"), "a+")
        results_file.write(
            "{},{}\n".format(relation["relation"], round(Precision1 * 100, 2))
        )
        results_file.flush()

        if "type" in relation:
            type_Precision1[relation["type"]].append(Precision1)
            data = load_file(PARAMETERS["dataset_filename"])
            type_count[relation["type"]].append(len(data))

    mean_p1 = statistics.mean(all_Precision1)
    mean_p10 = statistics.mean(all_Precision10)
    print("@@@ {} - mean P@1: {}".format(input_param["label"], mean_p1))
    print("@@@ {} - mean P@10: {}".format(input_param["label"], mean_p10))
    output_file.write("{},P@1,{}\n".format(input_param["label"], mean_p1))
    output_file.write("{},P@10,{}\n".format(input_param["label"], mean_p10))
    results_file.close()

    for t, l in type_Precision1.items():

        print(
            "@@@ ",
            input_param["label"],
            t,
            statistics.mean(l),
            sum(type_count[t]),
            len(type_count[t]),
            flush=True,
        )

        # output_file.write("{},{},{}\n".format(input_param["label"], t, statistics.mean(l)))
    output_file.close()

    return mean_p1, all_Precision1


def get_TREx_parameters(data_path_pre="data/"):
    relations = load_file(my_args.relations)
    data_path_pre += "{}/".format(my_args.data_dir)
    data_path_post = ".jsonl"
    return relations, data_path_pre, data_path_post


def get_GoogleRE_parameters():
    relations = [
        {
            "relation": "place_of_birth",
            "template": "[X] was born in [Y] .",
            "template_negated": "[X] was not born in [Y] .",
        },
        {
            "relation": "date_of_birth",
            "template": "[X] (born [Y]).",
            "template_negated": "[X] (not born [Y]).",
        },
        {
            "relation": "place_of_death",
            "template": "[X] died in [Y] .",
            "template_negated": "[X] did not die in [Y] .",
        },
    ]
    data_path_pre = "data/Google_RE/"
    data_path_post = "_test.jsonl"
    return relations, data_path_pre, data_path_post


def get_ConceptNet_parameters(data_path_pre="data/"):
    relations = [{"relation": "test"}]
    data_path_pre += "ConceptNet/"
    data_path_post = ".jsonl"
    return relations, data_path_pre, data_path_post


def get_Squad_parameters(data_path_pre="data/"):
    relations = [{"relation": "test"}]
    data_path_pre += "Squad/"
    data_path_post = ".jsonl"
    return relations, data_path_pre, data_path_post


def run_all_LMs(parameters):
    for ip in LMs:
        print(ip["label"])
        run_experiments(*parameters, input_param=ip, use_negated_probes=False)


if __name__ == "__main__":
    """
    print("1. Google-RE")
    parameters = get_GoogleRE_parameters()
    run_all_LMs(parameters)

    print("2. T-REx")
    """
    parameters = get_TREx_parameters()
    run_all_LMs(parameters)
    """
    print("3. ConceptNet")
    parameters = get_ConceptNet_parameters()
    run_all_LMs(parameters)

    print("4. SQuAD")
    parameters = get_Squad_parameters()
    run_all_LMs(parameters)
    """
