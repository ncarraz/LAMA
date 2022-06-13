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
    "facebook/bart-base":"bpe",
    "facebook/bart-large":"bpe",
}

class MaskedLM(Base_Connector):

    def __init__(self, args, vocab_subset = None):
        super().__init__()
        self.model_name = args.model_name
        self.tokenization = TOKENIZATION[self.model_name]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.__init_vocab() # Compatibility with existing code
        
        self.masked_model = AutoModelForMaskedLM.from_pretrained(self.model_name)
        self.mask = self.tokenizer.mask_token
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
                if value[0] == sep_token:  # if the token starts with a whitespace
                    value = value[1:]
                elif value not in special_tokens:
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
        output = self.tokenizer(sentences_list, padding=True, return_tensors="pt")
        masked_indices_list = np.argwhere(output.input_ids.numpy() == self.tokenizer.mask_token_id)[:,1]
        masked_indices_list = [[i] for i in masked_indices_list]

        with torch.no_grad():
            scores = self.masked_model(**output.to(self._model_device)).logits
            log_probs = F.log_softmax(scores, dim=-1).cpu()
        # second returned value is off for seq2seq
        return log_probs, list(output.input_ids.cpu().numpy()), masked_indices_list

    def get_contextual_embeddings(self, sentences_list, try_cuda=True):
        # Compatibility with existing code
        pass
