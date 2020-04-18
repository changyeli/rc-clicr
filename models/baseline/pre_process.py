import json
import re
import os
import sys
import tqdm
import pandas as pd


"""
Dataset pre-processing
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


def add_to_dataframe(file, outfile):
    """
    convert each record in the json file into dataframe row
    :param str file: the filename of the json file
    :param str outfile: the local file to save the dataframe
    """
    df = pd.DataFrame()
    file = file + "1.0.json"
    path = os.path.join("../../clicr", file)
    with open(path) as f:
        data = json.load(f)
        for p in tqdm.tqdm(data["data"]):
            source = p["source"]
            title = extract_context(p, "title")
            context = extract_context(p, "context")
            body = title + " " + context
            tokens = extract_entity(body)
            body = re.sub(r"[^\x00-\x7F]+", " ", body)
            qas = extract_context(p, "qas")
            for pairs in qas:
                query = pairs["query"]
                qid = pairs["id"]
                answers = [item["text"] for item in pairs["answers"]]
                cuis = [item["cui"] for item in pairs["answers"]]
                df = df.append({"source": source, "body": body, "tokens": tokens,
                                "qid": qid, "query": query, "answers": answers,
                                "cui": list(set(cuis))[0]},
                               ignore_index=True)
    print(df.head())
    print(df.shape)


if __name__ == '__main__':
    add_to_dataframe("train", "test")
