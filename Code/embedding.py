# ---------------------------------------------------------------Method 1-------------------------------------------------------------- #
'''
load RoBERTa from torch.hub
import torch

roberta_torch = torch.hub.load('pytorch/fairseq', 'roberta.large')
roberta_torch.eval()

sentence = "I Love RoBERTa!!! I Love Pytorch!!!"
Apply Byte-Pair Encoding to input text, tokens should be a tensor
tokens_torch = roberta_torch.encode(sentence)

Extract features from RoBERTa using BPE text
embedding_torch = roberta_torch.extract_features(tokens_torch, return_all_hiddens=True)[0]
'''

# ---------------------------------------------------------------Method 2-------------------------------------------------------------- #
'''
load RoBERTa from transformers, note it does not have .encode(), therefore we need RobertaTokenizer
import torch
from transformers import RobertaModel, RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
roberta_trans = RobertaModel.from_pretrained("roberta-large")

sentence = "I Love RoBERTa!!! I Love Pytorch!!!"
Apply Byte-Pair Encoding to input text with RobertaTokenizer, note that tokenizer.encode() returns to you a list, but we need our tokens to be a tensor
tokens_trans = torch.tensor([tokenizer.encode(sentence)])

Extract features from RobertaModel using BPE text
embedding_trans = roberta_trans.embeddings(tokens_trans)[0]
'''

import torch

class RobertaModel():
    def __init__(self):
        self.model = torch.hub.load('pytorch/fairseq', 'roberta.large')
        self.model.eval()

    def __call__(self, content):
        tokens = self.model.encode(content)
        embed = self.model.extract_features(tokens, return_all_hiddens=True)[0]
        return embed


if __name__ == '__main__':
    roberta = RobertaModel()
    encoding = roberta('content goes here')