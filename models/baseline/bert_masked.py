import torch
import sys
import os
import tqdm
import json
from datetime import datetime
from transformers import BertTokenizer, BertForMaskedLM
from utils import extract_context
from utils import clean_paragraph
from utils import generate_bert_format_qas
from utils import generate_bert_format_context


"""
Bert for Masked language model, with pre-processing
"""


def iterate_first_20_doc(infile):
    """
    iterate the first 20 doc from the input file,
    preprocess each doc, and
    :param infile: the filename of the json file
    :return:
    """
    read_file = infile + "1.0.json"
    path = os.path.join("../../clicr", read_file)
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # for reproducible results during evaluation
    model.eval()
    model.to("cuda")
    line = 0
    with open(path) as f:
        data = json.load(f)
        for p in tqdm.tqdm(data["data"]):
            title = extract_context(p, "title")
            context = extract_context(p, "context")
            title = clean_paragraph(title)
            context = clean_paragraph(context)
            # question and answers
            qas = extract_context(p, "qas")
            for pairs in qas:
                source = pairs["id"]
                question = pairs["query"]
                answers = [item["text"] for item in pairs["answers"]]
                for ans in answers:
                    query = generate_bert_format_qas(question, ans, tokenizer)
                    body = generate_bert_format_context(title, context)
                    body_tokens = tokenizer.tokenize(body)
                    query_tokens = tokenizer.tokenize(query)
                    tokens = query_tokens + body_tokens
                    segment_id = [0] * len(query_tokens) + [1] * len(body_tokens)
                    # keep only first 512 tokens
                    segment_id = segment_id[:512]
                    tokens = tokens[:512]
                    # make sure the embedding length == 512
                    assert len(tokens) == 512
                    # generate mask indexes and associated locations
                    index_ids = [0] * len(tokens)
                    for i, v in enumerate(tokens):
                        if tokens[i] == "[MASK]":
                            index_ids[i] = 1
                    index_loc = [i for i, v in enumerate(index_ids) if v == 1]
                    indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)
                    segment_tensor = torch.tensor([segment_id])
                    segment_tensor = segment_tensor.to("cuda")
                    tokens_tensor = torch.tensor([indexed_tokens])
                    tokens_tensor = tokens_tensor.to("cuda")
                    with torch.no_grad():
                        outputs = model(tokens_tensor, token_type_ids=segment_tensor)
                        predictions = outputs[0]
                    # get prediction token
                    pred_index = [torch.argmax(predictions[0, index]).item() for index in index_loc]
                    pred_text = [tokenizer.convert_ids_to_tokens(index) for index in pred_index]
                    # make sure every masked token get predictions
                    assert len(index_loc) == len(pred_text)
                    print(pred_text)
                    break
                break
            break
            line += 1


if __name__ == '__main__':
    iterate_first_20_doc("train")
