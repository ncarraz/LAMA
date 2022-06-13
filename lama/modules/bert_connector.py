import torch
import numpy as np
from lama.modules.base_connector import *
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM, AutoModelForSeq2SeqLM

TOKENIZATION = {
    "roberta-base":"bpe",
    "roberta-large":"bpe",
    "allenai/longformer-base-4096":"bpe",
    "allenai/longformer-large-4096":"bpe",
    "distilroberta-base":"bpe",
    "bert-base-cased":"wordpiece",
    "bert-large-cased":"wordpiece",
    "distilbert-base-cased":"wordpiece",
    "gpt2":"bpe",
    "xlnet-base-cased":"sentencepiece",
    "xlnet-large-cased":"sentencepiece",
    "facebook/bart-base":"bpe",
    "facebook/bart-large":"bpe",
    "t5-small":"sentencepiece",
    "t5-base":"sentencepiece",
    "t5-large":"sentencepiece"
}

LM_TYPE = {
    "roberta-base":"masked",
    "roberta-large":"masked",
    "allenai/longformer-base-4096":"masked",
    "allenai/longformer-large-4096":"masked",
    "distilroberta-base":"masked",
    "bert-base-cased":"masked",
    "bert-large-cased":"masked",
    "distilbert-base-cased":"masked",
    "gpt2":"causal",
    "xlnet-base-cased":"causal",
    "xlnet-large-cased":"causal",
    "facebook/bart-base":"masked",
    "facebook/bart-large":"masked",
    "t5-small":"seq2seq",
    "t5-base":"seq2seq",
    "t5-large":"seq2seq"
}

class Bert(Base_Connector):

    def __init__(self, args, vocab_subset = None):
        super().__init__()
        self.model_name = args.bert_model_name
        self.tokenization = TOKENIZATION[self.model_name]
        self.model_type = LM_TYPE[self.model_name]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.__init_vocab() # Compatibility with existing code
        
        if self.model_type == "causal":
            self.masked_model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.mask = self.tokenizer.eos_token # Compatibility with existing evaluation
        elif self.model_type == "masked":
            self.masked_model = AutoModelForMaskedLM.from_pretrained(self.model_name)
            self.mask = self.tokenizer.mask_token
        elif self.model_type == "seq2seq":
            self.masked_model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.mask = "<extra_id_0>" # TO DO: adapt to others, t5 only at this point
        self.masked_model.eval() # EVAL ONLY ?

    def __init_vocab(self):
        if self.tokenization in ["bpe", "sentencepiece"]: 
            # Convert vocabulary to BERT
            special_tokens = [self.tokenizer.bos_token, self.tokenizer.eos_token, self.tokenizer.unk_token,
                            self.tokenizer.sep_token, self.tokenizer.pad_token, self.tokenizer.cls_token,
                            self.tokenizer.mask_token]
            separator_tokens = {"bpe":"Ġ", "sentencepiece":"▁"}
            sep_token = separator_tokens[self.tokenization]
            converted_vocab = {}
            for w, i in self.tokenizer.vocab.items():
                value = w
                if value[0] == sep_token and value not in special_tokens:  # if the token starts with a whitespace
                    value = value[1:]
                else:
                    # this is subword information
                    value = "_{}_".format(value)

                if value in converted_vocab:
                    # print("WARNING: token '{}' is already in the vocab".format(value))
                    value = "{}_{}".format(value, i)
                converted_vocab[value] = i
        else:
            converted_vocab = self.tokenizer.vocab

        # Compatibility with existing code
        self.vocab = list(dict(sorted(converted_vocab.items(), key=lambda item: item[1])).keys())
        self.inverse_vocab = converted_vocab

    def get_id(self, string):
        if "bpe" in self.tokenization:
            string = " " + string
            
        tokenized_text = self.tokenizer.tokenize(string)
        indexed_string = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        return indexed_string

    def _cuda(self):
        self.masked_model.cuda()

    def get_batch_generation(self, sentences_list, logger= None, try_cuda=True):
        if not sentences_list:
            return None
        if try_cuda:
            self.try_cuda()

        # Compatibility with existing code
        sentences_list = [item for sublist in sentences_list for item in sublist]
        if self.model_type == "causal": # only consider left context
            sentences_list = [sentence.split(self.mask)[0] for sentence in sentences_list]
            
        output = self.tokenizer(sentences_list, padding=True, return_tensors="pt")
        
        if self.model_type == "causal":
            masked_indices_list = np.argmax(output.input_ids.numpy(), axis=1) - 1
        elif self.model_type == "masked":
            masked_indices_list = np.argwhere(output.input_ids.numpy() == self.tokenizer.mask_token_id)[:,1]
        elif self.model_type == "seq2seq":
            masked_indices_list = [1] * len(sentences_list)
        masked_indices_list = [[i] for i in masked_indices_list]

        with torch.no_grad():
            if self.model_type == "seq2seq":
                scores = self.masked_model.generate(output.input_ids.to(self._model_device), 
                                                    max_new_tokens=2, output_scores=True, return_dict_in_generate=True).scores
                scores = torch.stack(scores, dim=1)
            else:
                scores = self.masked_model(**output.to(self._model_device)).logits
            log_probs = F.log_softmax(scores, dim=-1).cpu()
        # seconde returned value is off for seq2seq
        return log_probs, list(output.input_ids.cpu().numpy()), masked_indices_list

    def get_contextual_embeddings(self, sentences_list, try_cuda=True):
        # Compatibility with existing code
        pass
