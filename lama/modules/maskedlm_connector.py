import torch
import numpy as np
from lama.modules.base_connector import *
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSeq2SeqLM

class MaskedLM(Base_Connector):

    def __init__(self, args, vocab_subset=None):
        super().__init__()
        self.model_name = args.model_name
        self.tokenization = TOKENIZATION[self.model_name]
        self.model_type = LM_TYPE[self.model_name]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._init_vocab() # Compatibility with existing code
        
        if self.model_type == "seq2seq":
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.mask = "<extra_id_0>" # for t5 only for now 
        elif self.model_type == "masked":
            self.mask = self.tokenizer.mask_token
            self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)
        self.model.eval() # EVAL ONLY ?

    def _cuda(self):
        self.model.cuda()
    
    def get_id(self, string):
        if "bpe" in self.tokenization:
            string = " " + string  
        tokenized_text = self.tokenizer.tokenize(string)
        indexed_string = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        return indexed_string

    def get_batch_generation(self, sentences_list, logger= None, try_cuda=True):
        if not sentences_list:
            return None
        if try_cuda:
            self.try_cuda()
        # Compatibility with existing code
        sentences_list = [item for sublist in sentences_list for item in sublist]
        input = self.tokenizer(sentences_list, padding=True, return_tensors="pt")
        masked_indices_list = np.argwhere(input.input_ids.numpy() == self.tokenizer.mask_token_id)[:,1]
        masked_indices_list = [[i] for i in masked_indices_list]
        with torch.no_grad():
            input['labels'] =  input['input_ids'] if 't5' in self.model.model_name else None
            scores = self.model(**input.to(self._model_device)).logits
            log_probs = F.log_softmax(scores, dim=-1).cpu()
        # second returned value is off for seq2seq
        return log_probs, list(input.input_ids.cpu().numpy()), masked_indices_list

    def get_contextual_embeddings(self, sentences_list, try_cuda=True):
        # Compatibility with existing code
        pass
