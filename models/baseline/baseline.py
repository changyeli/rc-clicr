import torch
import sys
import json
import spacy
import pandas as pd
import gc
import tqdm
from scipy.stats import describe
from datetime import datetime
from transformers import BertTokenizer, BertForMaskedLM


"""
The baseline model with:
    - Bert pre-trained model
    - clustering model
    - BiLSTM model
"""


# install spacy model
# python -m spacy download en_trf_bertbaseuncased_lg
# python -m spacy download en_core_web_lg
is_using_gpu = spacy.prefer_gpu()
if is_using_gpu:
    torch.set_default_tensor_type("torch.cuda.FloatTensor")


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
    total_lines = sum(1 for line in open(input_file, "r"))
    # predict answers per qas
    with open(input_file, "r") as f:
        for line in tqdm.tqdm(f, total=total_lines):
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


def calculate_sim_score(infile, method):
    """
    read the file and use the data to run clustering models with similairty score
    read only the first 500 records
    :param str input_file: the file to read
    :param str method: the embedding method
    :return: the dataframe with similarity score
    :rtype: pandas.DataFrame
    """
    if method == "bert":
        df_with_bert = pd.DataFrame()
        nlp_bert = spacy.load("en_trf_bertbaseuncased_lg")
        total_lines = sum(1 for line in open(infile, "r"))
        with open(infile, "r") as f:
            for line in tqdm.tqdm(f, total=total_lines):
                data = json.loads(line)
                true_content = data["query"]
                pos_content = data["possible_qas"]
                content_1 = nlp_bert(true_content)
                content_2 = nlp_bert(pos_content)
                score = content_1[0].similarity(content_2[0])
                print("similarity score: {}".format(score))
                df_with_bert = df_with_bert.append({"source": data["source"],
                                                      "body": data["body"],
                                                      "query": data["query"],
                                                      "answer": data["answer"],
                                                      "candidate": data["poss_answer"],
                                                      "score": score},
                                                     ignore_index=True)
        return df_with_bert
    elif method == "spacy":
        df_with_spacy = pd.DataFrame()
        nlp_spacy = spacy.load("en_core_web_lg")
        total_lines = sum(1 for line in open(infile, "r"))
        with open(infile, "r") as f:
            for line in tqdm.tqdm(f, total=total_lines):
                data = json.loads(line)
                true_content = data["query"]
                pos_content = data["possible_qas"]
                content_1 = nlp_spacy(true_content)
                content_2 = nlp_spacy(pos_content)
                score = content_1[0].similarity(content_2[0])
                print("similarity score: {}".format(score))
                df_with_spacy = df_with_spacy.append({"source": data["source"],
                                                      "body": data["body"],
                                                      "query": data["query"],
                                                      "answer": data["answer"],
                                                      "candidate": data["poss_answer"],
                                                      "score": score},
                                                     ignore_index=True)
        return df_with_spacy
    else:
        raise ValueError("Please double check the embedding method.")


def describe_dataset(df, data_type, method):
    """
    describe score distribution
    :param pandas.DataFrame df: dataframe with similarity score
    :param str data_type: the type of data for df
    :param str method: the embedding method
    :return:
    """
    print("similarity score with bert in {} set with {}.".format(data_type, method))
    print(describe(df["score"]))
    outfile = "../../clicr/" + data_type + "_" + method + ".pickle"
    df.to_pickle(outfile)


def ad_hoc_analysis(infile):
    """
    ad-hoc analysis for spacy methods
    :param str input_file: the file to read
    :return:
    """
    if "train" in infile:
        bert_df = calculate_sim_score(infile, "bert")
        describe_dataset(bert_df, "train", "bert")
        spacy_df = calculate_sim_score(infile, "spacy")
        describe_dataset(spacy_df, "train", "spacy")
    elif "test" in infile:
        bert_df = calculate_sim_score(infile, "bert")
        describe_dataset(bert_df, "test", "bert")
        spacy_df = calculate_sim_score(infile, "spacy")
        describe_dataset(spacy_df, "test", "spacy")
    elif "dev" in infile:
        bert_df = calculate_sim_score(infile, "bert")
        describe_dataset(bert_df, "dev", "bert")
        spacy_df = calculate_sim_score(infile, "spacy")
        describe_dataset(spacy_df, "dev", "spacy")


if __name__ == '__main__':
    start_time = datetime.now()
    ad_hoc_analysis("../../clicr/train_norm.jsonl")
    gc.collect()
    ad_hoc_analysis("../../clicr/test_norm.jsonl")
    gc.collect()
    ad_hoc_analysis("../../clicr/dev_norm.jsonl")
    gc.collect()
    sys.stdout.write("total running time: {}".format(datetime.now() - start_time))
