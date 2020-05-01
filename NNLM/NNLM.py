import argparse
import os
from collections import Counter
import numpy as np
from describe_data import *
from evaluate import evaluate, print_scores, f1_score
from text import VocabBuild
from util import load_json, save_json, random_instance_from_list, cosines
import re
import tensorflow as tf
import tensorflow_hub as hub

def read_concepts(text):
    """
    Reading concepts from the context using the BEG, END indicator
    :param: string: the context
    :return: the concepts list 
    """
    concept_list = []
    inside = False
    for w in text.split():
        w_stripped = w.strip()
        if w_stripped.startswith("BEG__") and w_stripped.endswith("__END"):
            concept = w_stripped.split("_")[2]
            concept_list.append([concept])
        elif w_stripped.startswith("BEG__"):
            inside = True
            concept = [w_stripped.split("_", 2)[-1]]
        elif w_stripped.endswith("__END"):
            concept.append(w_stripped.rsplit("_", 2)[0])
            concept_list.append(concept)
            inside = False
        else:
            if inside:
                concept.append(w_stripped)
            else:
                continue
    return concept_list


def embedding_correlation(dataset, embedder):
    """
    Answering the query by comparing the Pearson correlation of the embedding 
    vectors of the query and the concepts list
    :param dataset: the json data file
    :param embedder: pre-trained text embedder
    :return: prediction of the query  
    """
    def all_concepts(text):
        """
        Find all concepts within the context
        :param text: contains concepts annotated as 'w_i BEG__w_k w_j__END w_l',
        where 'w_k w_k' is a concept.
        :return: a concept
        """
        concept_list = read_concepts(text)
        all_concept = []
        for i in concept_list:
            tmp = " ".join(i)
            all_concept.append(tmp)
        return all_concept

    data = dataset[DATA_KEY]
    predictions = {}
    i = 0
    for datum in data:
        title_and_passage = datum[DOC_KEY][TITLE_KEY] + "\n" + datum[DOC_KEY][CONTEXT_KEY]
        for qa in datum[DOC_KEY][QAS_KEY]:
            id = qa[ID_KEY]
            
            query = qa['query'].replace("BEG__", "").replace("__END", "").replace("@placeholder", "")
            concepts = all_concepts(title_and_passage)
            concepts.append(query)
            concepts = np.array(concepts)
            X = embedder(concepts)
            corr = np.abs(np.corrcoef(X.numpy())[-1,:])
            predictions[id] = concepts[np.argmax(corr[:-1])]
        #i +=1
        #if i > 19:
            #break
    return predictions


if __name__ == '__main__':
    EMBED_URL = "https://tfhub.dev/google/nnlm-en-dim128/2"  # The embedding model.
    print("Getting embedding model...", end='')
    embedder = hub.KerasLayer(EMBED_URL, dtype=tf.string, trainable=False) # The Layer, which does the transformation.
    print("Done")
    
    test_file = "../../data/train1.0.json"
    dataset = load_json(test_file)
    predictions_concepts = embedding_correlation(dataset, embedder)
    scores_concepts = evaluate(dataset, predictions_concepts)   
    print("Train dataset:")
    print(scores_concepts)
    
    test_file = "../../data/test1.0.json"
    dataset = load_json(test_file)
    predictions_concepts = embedding_correlation(dataset, embedder)
    scores_concepts = evaluate(dataset, predictions_concepts)   
    print("Test dataset:")
    print(scores_concepts)

    test_file = "../../data/dev1.0.json"
    dataset = load_json(test_file)
    predictions_concepts = embedding_correlation(dataset, embedder)
    scores_concepts = evaluate(dataset, predictions_concepts)   
    print("Dev dataset:")
    print(scores_concepts)
    
# =============================================================================
# test_file = "../../data/dev1.0.json"
# dataset = load_json(test_file)
# EMBED_URL = "https://tfhub.dev/google/nnlm-en-dim128/2"  # The embedding model.
# print("Getting embedding model...", end='')
# embedder = hub.KerasLayer(EMBED_URL, dtype=tf.string, trainable=False) # The Layer, which does the transformation.
# print("Done")
# 
# 
# predictions_concepts = embedding_correlation(dataset, embedder)
# scores_concepts = evaluate(dataset, predictions_concepts)   
# scores_concepts
# =============================================================================
