import sys
import re
import string


"""
utility functions
"""


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


def extract_entity(paragraph):
    """
    extract all entities from the paragraph, BEG__***__END
    return all extracted entities
    :param str paragraph: the paragraph needs to be processed
    :return: a list of entities
    :rtype: list
    """
    tokens = re.findall(r"BEG__(.*?)__END", paragraph)
    tokens = [item.lower() for item in tokens]
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
    :param str title: the cleaned title of the clinical reports
    :param str context: the cleaned context of the clinical reports
    :return: bert formatted full body and segment id list
    :rtype: str
    """
    sentences = title + " " + context
    sentences = "[SEP] " + sentences
    return sentences


def generate_context(title, context):
    """
    generate the norm format given title and context
    :param str title: the cleaned title of the clinical reports
    :param str context: the cleaned context of the clinical reports
    :return: the full context for the question
    :rtype: str
    """
    sentences = title + " " + context
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
    :return: the bert format question
    :rtype: str
    """
    question = clean_paragraph(question)
    question = "[CLS] " + question
    tokens = tokenizer.tokenize(answer)
    pattern = "[MASK] " * len(tokens)
    question = re.sub(r"@placeholder", pattern, question)
    return question