import json
import re
import os
import sys
import tqdm
from datetime import datetime
from transformers import BertTokenizer

"""
Dataset pre-processing for Bert format
"""


def extract_entity(paragraph):
    """
    extract all entities from the paragraph, BEG__***__END
    return all extracted entities
    :param str paragraph: the paragraph needs to be processed
    :return: a list of entities
    :rtype: list
    """
    tokens = re.findall(r"BEG__(.*?)__END", paragraph)
    return tokens


def clean_paragraph(paragraph):
    """
    remove all BEG__ and __END, newlines, lower the characters,
    remove stopwords and non-ASCII characters
    :param str paragraph: the concated title and context
    :return: cleaned title and context
    :rtype: str
    """
    paragraph = re.sub(r"[^\x00-\x7F]+", " ", paragraph)
    paragraph = re.sub(r"BEG__", "", paragraph)
    paragraph = re.sub(r"__END", "", paragraph)
    paragraph = re.sub(r"\s+", " ", paragraph)
    paragraph = paragraph.lower()
    return paragraph


def generate_bert_format_context(title, context):
    """
    generate the bert format given title and context
    :param str title: the title of the clinical reports
    :param str context: the context of the clinical reports
    :return: bert formatted full body and segment id list
    :rtype: tuple
    """
    sentences = title + " " + context
    sentences = "[SEP] " + sentences
    return sentences


def generate_bert_format_qas(question, answer, tokenizer):
    """
    generate the bert format for a pair of question and answer pair
    :param str question: the question from raw dataset
    :param str answer: the answer from raw dataset
    :param transformers.BertTokenizer: the bert tokenizer
    :param str data_type: the type of the data
                          if the dataset is training set, then replace the answers
                            to the @placehoder
                          else, put the [MASK] tokens in the question
    :return: the bert format question, question for training and index ids
    :rtype: tuple
    """
    question = clean_paragraph(question)
    question = "[CLS] " + question
    tokens = tokenizer.tokenize(answer)
    question_for_train = re.sub(r"@placeholder", answer.lower(), question)
    pattern = "[MASK] " * len(tokens)
    question = re.sub(r"@placeholder", pattern, question)
    return question, question_for_train


def extract_context(doc, key):
    """
    from a dictionary document doc, retrieve the content under key
    :param dic doc: the dictionary for the document from the json file
    :param key: the property (i.e., key) from the dictionary
    :return: the content for the property
    :rtype: str/list
    """
    doc = doc["document"]
    try:
        doc = doc[key]
        return doc
    except KeyError:
        sys.stdout.write("No such key exists, please double check!")


def add_to_dataframe(file):
    """
    convert each record in the json file into jsonl
    only save the frist 200 records
    :param str file: the filename of the json file
    """
    read_file = file + "1.0.json"
    path = os.path.join("../../clicr", read_file)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    outfile = file + "_cleaned.jsonl"
    outpath = os.path.join("../../clicr", outfile)
    outfile_train = file + "_cleaned_for_train.jsonl"
    outpath_train = os.path.join("../../clicr", outfile_train)
    # line counter
    line = 0
    with open(path) as f:
        data = json.load(f)
        for p in tqdm.tqdm(data["data"]):
            # context body
            title = extract_context(p, "title")
            context = extract_context(p, "context")
            title = clean_paragraph(title)
            context = clean_paragraph(context)
            # question and answers
            qas = extract_context(p, "qas")
            for pairs in qas:
                # question id
                source = pairs["id"]
                question = pairs["query"]
                answers = [item["text"] for item in pairs["answers"]]
                for ans in answers:
                    # save one copy for further training process
                    if file == "train":
                        question, question_for_train = generate_bert_format_qas(question,
                                                                                ans, tokenizer)
                        body = generate_bert_format_context(title, context)
                        len_ques = len(tokenizer.tokenize(question))
                        len_body = len(tokenizer.tokenize(body))
                        segment_id = [0] * len_ques + [1] * len_body
                        piece = {"source": source, "body": body, "segment_ids": segment_id,
                                 "query": question, "answers": answers}
                        with open(outpath, "a") as outF:
                            json.dump(piece, outF)
                            outF.write("\n")
                        piece = {"source": source, "body": body, "segment_ids": segment_id,
                                 "query": question_for_train, "answers": answers}
                        with open(outpath_train, "a") as outF:
                            json.dump(piece, outF)
                            outF.write("\n")
                    else:
                        question, _ = generate_bert_format_qas(question, ans, tokenizer)
                        body = generate_bert_format_context(title, context)
                        len_ques = len(tokenizer.tokenize(question))
                        len_body = len(tokenizer.tokenize(body))
                        segment_id = [0] * len_ques + [1] * len_body
                        piece = {"source": source, "body": body, "segment_ids": segment_id,
                                 "query": question, "answers": answers}
                        with open(outpath, "a") as outF:
                            json.dump(piece, outF)
                            outF.write("\n")
                    line += 1
            if line > 200:
                break


if __name__ == '__main__':
    start_time = datetime.now()
    sys.stdout.write("=============================\n")
    sys.stdout.write("Pre-processing training set\n")
    add_to_dataframe("train")
    sys.stdout.write("=============================\n")
    sys.stdout.write("Pre-processing test set\n")
    add_to_dataframe("test")
    sys.stdout.write("=============================\n")
    sys.stdout.write("Pre-processing development set\n")
    add_to_dataframe("dev")
    sys.stdout.write("=============================\n")
    sys.stdout.write("total running time: {}".format(datetime.now() - start_time))
