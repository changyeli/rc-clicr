import torch
import os
import tqdm
import json
import sys
import pandas as pd
import logging
from datetime import datetime
from transformers import BertTokenizer, BertForMaskedLM
from utils import extract_context
from utils import clean_paragraph
from utils import generate_bert_format_qas
from utils import generate_bert_format_context

"""
Bert for Masked language model, with pre-processing
"""


def iterate_first_n_doc(infile, n):
    """
    iterate the first 50 doc from the input file,
    preprocess each doc, and make prediction
    write predicted tokens in to local file
    :param str infile: the type of input json file
    :param int n: number of docs to iterate
    """
    read_file = infile + "1.0.json"
    path = os.path.join("../../clicr", read_file)
    outfile = infile + "_pred.json"
    outpath = os.path.join("../../clicr", outfile)
    model = BertForMaskedLM.from_pretrained("bert-large-uncased-whole-word-masking")
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking")
    # for reproducible results during evaluation
    model.eval()
    model.to("cuda")
    line = 0
    with open(path) as f:
        data = json.load(f)
        for p in tqdm.tqdm(data["data"]):
            title = extract_context(p, "title")
            context = extract_context(p, "context")
            # remove punctuations
            title = clean_paragraph(title)
            context = clean_paragraph(context)
            # question and answers
            qas = extract_context(p, "qas")
            for pairs in qas:
                source = pairs["id"]
                question = pairs["query"]
                answers = [item["text"] for item in pairs["answers"]]
                answers = [item.lower() for item in answers]
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
                    # padding tokens to 512
                    if len(tokens) < 512:
                        num_pad = 512 - len(tokens)
                        padding = ["[PAD]"] * num_pad
                        tokens += padding
                        segment_id = segment_id + [1] * num_pad
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
                    # human readable answer
                    fine_text = " ".join([x for x in pred_text])
                    fine_text = fine_text.replace(" ##", "")
                    piece = {"qid": source,
                             "true_answer": ans.lower(),
                             "pred_answer": fine_text,
                             "EM": ans.lower() == fine_text,
                             "answers": answers}
                    with open(outpath, "a") as outF:
                        json.dump(piece, outF)
                        outF.write("\n")
            line += 1
            if line > n:
                break


def calculate_em_answer_level(infile, n):
    """
    calculate the EM shares given an input file on answer level
    # TODO: calcualte partial match between two strings
    :param str infile: the type of input json file
    :param int n: number of docs to iterate
    """
    outfile = infile + "_pred.json"
    outpath = os.path.join("../../clicr", outfile)
    sys.stdout.write("answer level calculation\n")
    if os.path.isfile(outpath):
        em = 0
        count = 0
        find_true_tokens = 0
        with open(outpath, "r") as f:
            for line in f:
                content = json.loads(line)
                if content["EM"]:
                    em += 1
                if content["pred_answer"] in content["answers"]:
                    find_true_tokens += 1
                count += 1
        sys.stdout.write("EM count: {}\n".format(em))
        sys.stdout.write("# of tokens are true answers count: {}\n".format(find_true_tokens))
        sys.stdout.write("total q&a pairs: {}\n".format(count))
    else:
        iterate_first_n_doc(infile, n)
        calculate_em_answer_level(infile, n)


def calculate_em_question_level(infile, n):
    """
    calculate exact match on question level, with first n docs
    :param infile: the type of input data
    :param n: the number of first n focs
    """
    outfile = infile + "_pred.json"
    outpath = os.path.join("../../clicr", outfile)
    sys.stdout.write("question level calculation\n")
    if os.path.isfile(outpath):
        em = 0
        find_true_tokens = 0
        df = pd.read_json(outpath, lines=True)
        qids = df["qid"].unique()
        for ids in qids:
            temp = df.loc[df["qid"] == ids]
            em_count = temp.loc[temp["EM"] == True]
            if em_count.shape[0] > 0:
                em += 1
            answer_list = temp["pred_answer"].unique()
            true_answer_list = temp["answers"].values.tolist()[0]
            common = any(item in answer_list for item in true_answer_list)
            if common:
                find_true_tokens += 1
        sys.stdout.write("EM count: {}\n".format(em))
        sys.stdout.write("# of tokens are true answers count: {}\n".format(find_true_tokens))
        sys.stdout.write("total questions: {}\n".format(len(qids)))
    else:
        iterate_first_n_doc(infile, n)
        calculate_em_question_level(infile, n)


if __name__ == '__main__':
    log = open("bert_masked.log", "a")
    sys.stdout = log
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        filemode="a", level=logging.INFO, filename="bert_masked.log")
    start_time = datetime.now()
    n = 20
    sys.stdout.write("============== {} =============\n".format("train"))
    calculate_em_answer_level("train", n)
    calculate_em_question_level("train", n)
    sys.stdout.write("============== {} =============\n".format("test"))
    calculate_em_answer_level("test", n)
    calculate_em_question_level("test", n)
    sys.stdout.write("============== {} =============\n".format("dev"))
    calculate_em_answer_level("dev", n)
    calculate_em_question_level("dev", n)
    sys.stdout.write("total running time: {}\n".format(datetime.now() - start_time))
