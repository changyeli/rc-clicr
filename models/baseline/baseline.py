import torch
import os
import torch.nn as nn
import logging
import sys
from gensim.models import KeyedVectors
from datetime import datetime
from transformers import BertTokenizer, BertForMaskedLM


"""
The baseline model with Bert pre-trained model
"""


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    question = "[CLS] [MASK] from amniotic band disruption is a possibility"
    tokens = tokenizer.tokenize(question)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [0] * len(tokens)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensor = torch.tensor([segment_ids])
    # tokens_tensor = tokens_tensor.to(device)
    # segments_tensor = segments_tensor.to(device)
    tokens_tensor = tokens_tensor.cuda()
    segments_tensor = segments_tensor.cuda()
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    model.cuda()
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensor)
        pred = outputs[0]
    pred_index = torch.argmax(pred[0, 0]).item()
    pred_token = tokenizer.convert_ids_to_tokens(pred_index)[0]
    print("Predicted tokens: {}".format(pred_token))
