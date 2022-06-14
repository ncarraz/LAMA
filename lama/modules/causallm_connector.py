# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from lama.modules.base_connector import *


class CausalLM(Base_Connector):
    def __init__(self, args):
        super().__init__()

        self.model_name = args.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})
        self.tokenizer.add_special_tokens({'mask_token': "[MASK]"})
        self.tokenization = TOKENIZATION[self.model_name]
        self.mask = self.tokenizer.mask_token

        if self.model_name == "transfo-xl-wt103":
            self.vocab = list(self.tokenizer.idx2sym)
            self._init_inverse_vocab()
        else:
            self._init_vocab()

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.eval()

    def _cuda(self):
        self.model.cuda()
    
    def get_id(self, string):
        if "bpe" in self.tokenization:
            string = " " + string
        tokenized_text = self.tokenizer.tokenize(string)
        indexed_string = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        return indexed_string

    def get_batch_generation(self, sentences_list, logger=None, try_cuda=True):
        if try_cuda:
            self.try_cuda()

        # Replace the added [MASK] token with EOS token to make embeddings work
        sentences_list = [self.tokenizer.eos_token + " " + item for sublist in sentences_list for item in sublist]
        src_tensor = self.tokenizer(sentences_list, padding=True, return_tensors="pt").input_ids
        dest_tensor = self.tokenizer(sentences_list).input_ids
        masked_indices_list = np.argwhere(src_tensor.numpy() == self.tokenizer.mask_token_id)[:,1] - 1
        masked_indices_list = [[i] for i in masked_indices_list]
        src_tensor = torch.where(src_tensor == self.tokenizer.mask_token_id, self.tokenizer.eos_token_id, src_tensor)
        src_tensor = src_tensor[:,:-1]
        
        token_ids_list = [tokens[1:] for tokens in dest_tensor]
        token_ids_list = [[self.tokenizer.eos_token_id if i == self.tokenizer.mask_token_id else i for i in l] for l in token_ids_list]
        with torch.no_grad():
            log_probs = self.model(src_tensor.to(self._model_device))
            if self.model_name == "transfo-xl-wt103":
                log_probs = log_probs.prediction_scores.cpu()
            else:
                log_probs = log_probs.logits.cpu()

        return log_probs, token_ids_list, masked_indices_list

    def get_contextual_embeddings(self, batched_sentence_list):
        pass
