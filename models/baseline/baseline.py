import torch
import sys
import json
from gensim.models import KeyedVectors
from datetime import datetime
from transformers import BertTokenizer, BertForMaskedLM


"""
The baseline model with Bert pre-trained model
"""


def bert_baseline(input_file):
    """
    run the pre-trained bert masked lm for prediction
    :param str input_file: the file to read
    :return:
    """
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    model.to("cuda")
    # for reproducible results during evaluation
    model.eval()
    # predict answers per qas
    with open(input_file, "r") as f:
        for line in f:
            data = json.loads(line)
            query = data["query"]
            body = data["body"]
            body_seg_ids = data["segment_ids"]
            context = query + body
            tokens = tokenizer.tokenize(context)
            index_ids = [i for i, v in enumerate(tokens) if v == "[MASK]"]
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)
            tokens_tensor = torch.tensor([indexed_tokens])
            segments_tensor = torch.tensor([body_seg_ids])
            # send to GPU
            tokens_tensor = tokens_tensor.to("cuda")
            segments_tensor = segments_tensor.to("cuda")
            with torch.no_grad():
                outputs = model(tokens_tensor, token_type_ids=segments_tensor)
                predictions = outputs[0]
            # get prediction token
            for item in index_ids:
                pred_index = torch.argmax(predictions[0, item]).item()
                pred_token = tokenizer.convert_ids_to_tokens([pred_index])[0]
                print("predicted token: ".format(pred_token))
            break


if __name__ == '__main__':
    bert_baseline("../../clicr/train_cleaned.jsonl")
