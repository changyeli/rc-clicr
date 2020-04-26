import json
import re
import os
import sys
import tqdm
from nltk.tokenize import sent_tokenize
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


def generate_bert_format_context(title, context, tokenizer):
    """
    generate the bert format given title and context
    :param str title: the title of the clinical reports
    :param str context: the context of the clinical reports
    :param transformers.BertTokenizer: the bert tokenizer
    :return: bert formatted full body and segment id list
    :rtype: tuple
    """
    title = clean_paragraph(title)
    context = clean_paragraph(context)
    sentences = sent_tokenize(context)
    sentences = [title] + sentences
    segment_ids = []
    format = ""
    for index, item in enumerate(sentences):
        format = format + " [SEP] " + item
        new_sen = " [SEP] " + item
        segment_ids = segment_ids + [index] * (len(tokenizer.tokenize(new_sen)) + 1)
    return format, segment_ids


def generate_bert_format_qas(question, answer, tokenizer, data_type):
    """
    generate the bert format for a pair of question and answer pair
    :param str question: the question from raw dataset
    :param str answer: the answer from raw dataset
    :param transformers.BertTokenizer: the bert tokenizer
    :param str data_type: the type of the data
                          if the dataset is training set, then replace the answers
                            to the @placehoder
                          else, put the [MASK] tokens in the question
    :return: the bert format question and the segment id
    :rtype: tuple
    """
    question = clean_paragraph(question)
    question = "[CLS] " + question
    tokens = tokenizer.tokenize(answer)
    if data_type == "train":
        question = re.sub(r"@placeholder", answer, question)
        tokens = tokenizer.tokenize(question)
        indexes = [i for i in range(len(tokens)) if tokens[i] == "[MASK]"]
        index_ids = [0] * len(tokens)
        for i, value in enumerate(index_ids):
            if i in indexes:
                index_ids[i] = 1
        return question, index_ids
    else:
        pattern = "[MASK] " * len(tokens)
        question = re.sub(r"@placeholder", pattern, question)
        tokens = tokenizer.tokenize(question)
        indexes = [i for i in range(len(tokens)) if tokens[i] == "[MASK]"]
        index_ids = [0] * len(tokens)
        for i, value in enumerate(index_ids):
            if i in indexes:
                index_ids[i] = 1
        return question, index_ids


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
    convert each record in the json file into DataFrame row
    :param str file: the filename of the json file
    """
    read_file = file + "1.0.json"
    path = os.path.join("../../clicr", read_file)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    outfile = file + "_cleaned.jsonl"
    outpath = os.path.join("../../clicr", outfile)
    with open(path) as f:
        data = json.load(f)
        for p in tqdm.tqdm(data["data"]):
            # document id
            source = p["source"]
            # context body
            title = extract_context(p, "title")
            context = extract_context(p, "context")
            title = clean_paragraph(title)
            context = clean_paragraph(context)
            body, segment_id = generate_bert_format_context(title, context, tokenizer)
            # question and answers
            qas = extract_context(p, "qas")
            for pairs in qas:
                question = pairs["query"]
                answers = [item["text"] for item in pairs["answers"]]
                for ans in answers:
                    label, index_ids = generate_bert_format_qas(question, ans,
                                                                tokenizer, file)
                    piece = {"source": source, "body": body, "segment_ids": segment_id,
                             "query": label, "answers": answers, "index_ids": index_ids}
                    with open(outpath, "a") as outF:
                        json.dump(piece, outF)
                        outF.write("\n")


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
