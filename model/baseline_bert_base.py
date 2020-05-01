#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 18:32:59 2020

@author: keelia
"""

import logging
import json

from nltk.tokenize import RegexpTokenizer
from string import punctuation
import torch
from transformers import BertTokenizer, BertForMaskedLM
from biobert_embedding.embedding import BiobertEmbedding
from scipy import spatial
import pandas as pd

DATA_KEY = "data"
VERSION_KEY = "version"
DOC_KEY = "document"
QAS_KEY = "qas"
ANS_KEY = "answers"
TXT_KEY = "text"
ORIG_KEY = "origin"
ID_KEY = "id"
TITLE_KEY = "title"
CONTEXT_KEY = "context"
SOURCE_KEY = "source"
QUERY_KEY = "query"
CUI_KEY = "cui"

PLACEHOLDER_KEY = "@placeholder"

def load_json(filename):
    with open(filename) as in_f:
        return json.load(in_f)

def to_entities(text):
    """
    Text includes entities marked as BEG__w1 w2 w3__END. Transform to a single entity @entityw1_w2_w3.
    """
    word_list = []
    inside = False
    for w in text.split():
        w_stripped = w.strip()
        if w_stripped.startswith("BEG__") and w_stripped.endswith("__END"):
            concept = [w_stripped.split("_")[2]]
            word_list.append("@entity" + "_".join(concept))
            if inside:  # something went wrong, leave as is
                logging.info("Inconsistent markup.")
        elif w_stripped.startswith("BEG__"):
            assert not inside
            inside = True
            concept = [w_stripped.split("_", 2)[-1]]
        elif w_stripped.endswith("__END"):
            if not inside:
                return None
            assert inside
            concept.append(w_stripped.rsplit("_", 2)[0])
            word_list.append("@entity" + "_".join(concept))
            inside = False
        else:
            if inside:
                concept.append(w_stripped)
            else:
                word_list.append(w_stripped)

    return " ".join(word_list)


def load_data(in_file, max_example=None):

    documents = []
    questions = []
    answers = []
    answersets = []
    ids = []  # [(qa_id: entity_dict)]
    sources = []
    num_examples = 0
    num_all = 0
    dataset = load_json(in_file)

    for datum in dataset[DATA_KEY]:
        document = to_entities(datum[DOC_KEY][TITLE_KEY] + " " + datum[DOC_KEY][CONTEXT_KEY])
        document = document.lower()
        source = datum[SOURCE_KEY]
           
        assert document
        
        for qa in datum[DOC_KEY][QAS_KEY]:
            num_all += 1
            question = to_entities(qa[QUERY_KEY]).lower()
            assert question
        
            answer = ""
            answerset = []
            for ans in qa[ANS_KEY]:
                if ans[ORIG_KEY] == "dataset":
                    answer = ("@entity" + "_".join(ans[TXT_KEY].split())).lower()
                answerset.append([ans[TXT_KEY],ans[ORIG_KEY]])
            assert answer
            assert answerset
            
            ids.append(qa[ID_KEY])
            documents.append(document)
            questions.append(question)
            answers.append(answer)
            answersets.append(answerset)
            sources.append(source)
            num_examples += 1
            if (max_example is not None) and (num_examples >= max_example):
                break

        if (max_example is not None) and (num_examples >= max_example):
            break
        
    return documents, questions, answers, answersets, ids, sources

 
def entites(documents, sources):
    
    tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
    
    doc_dict = dict(zip(sources, documents)) 
    tokens_dict = {}
    entities_dict = {}
    
    for key, doc in doc_dict.items():
        
        tokens = [w for w in tokenizer.tokenize(doc) if w not in punctuation]
        entities = [w for w in tokens if w.startswith("@entity")]
        
        tokens_dict.update({key: tokens})
        entities_dict.update({key: entities})
    
    corpus = [w for w in tokens_dict.values()]
    tokens_lst = [tokens_dict[key] for key in sources] 
    entities_lst = [entities_dict[key] for key in sources] 
    
    return corpus, tokens_lst, entities_lst


def predict_from_pretirain_bert(infile, outfile):
    
    documents, questions, answers, answersets, ids, sources = load_data(infile,
                                                                        max_example=None)    
    d_corpus, d_tokens_lst, d_entities_lst = entites(documents, sources)
    
    biobert = BiobertEmbedding()
    
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Load pre-trained model (weights)
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    model.eval()
        
    predicted_token_lst = []
    document_ids = set()
    
    for i in range(len(documents)):
        
        document_ids.add(sources[i])
        
        if len(document_ids) > 20:
            print('Finish first 20 documents')
            break
            
        new_entities = [w.replace("@entity",'').replace("_",' ') for w in d_entities_lst[i]]
        tokenizer.add_tokens(new_entities)
        model.resize_token_embeddings(len(tokenizer))
    
        sent = documents[i].split('.')               
        text = questions[i].replace("@placeholder",answers[i])
        
        query_embedding = biobert.sentence_vector(text.replace("@entity",''))
        
        similarity = []
        for s in sent:
            sentence_embedding = biobert.sentence_vector(s.replace("@entity",''))    
            similarity.append(1 - spatial.distance.cosine(query_embedding, sentence_embedding))
        
        res = sorted(range(len(similarity)), key = lambda sub: similarity[sub])[-5:] 
        
        text_masked = '[CLS]' + questions[i].replace("@placeholder",'[MASK]').replace("@entity",'').replace("_",' ') 
    
    
        for idx in res:
            text_masked = text_masked + '[SEP]' + sent[idx].replace("@entity",'').replace("_",' ')
        text_masked = text_masked + '[SEP]'
        
        tokenized_text = tokenizer.tokenize(text_masked)
        clean_tokenized_text = [t for t in tokenized_text if t not in punctuation]
        masked_index = clean_tokenized_text.index('[MASK]')
        indexed_tokens = tokenizer.convert_tokens_to_ids(clean_tokenized_text)
        segments_ids = [0]*clean_tokenized_text.index('[SEP]') + [1]*(len(clean_tokenized_text)-clean_tokenized_text.index('[SEP]'))
        
        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
            
        # Predict tokens
        with torch.no_grad():
            outputs = model(tokens_tensor, token_type_ids=segments_tensors)
            predictions = outputs[0]
        
        answer_candidates = [tokenizer.get_vocab()[e] for e in new_entities]
        answer_candidates_dict = dict(zip(range(len(answer_candidates)),answer_candidates))
        predicted_index = torch.argmax(predictions[0, masked_index ,answer_candidates]).item()
        predicted_token = tokenizer.convert_ids_to_tokens([answer_candidates_dict[predicted_index]])[0]
        predicted_token_lst.append(predicted_token)
        #print(predicted_token)
        if i%20 == 0:
            print('Finish %d Queries'%(i))
            

            
    result = pd.DataFrame({'Query':ids[:len(predicted_token_lst)],
                           'True Answer': [w.replace("@entity",'').replace("_",' ') for w in answers[:len(predicted_token_lst)]],
                           'Pred Answer': predicted_token_lst,
                           'True Answer set': answersets[:len(predicted_token_lst)]})
    result.to_csv(outfile)
    
    AM=set()
    for i in range(len(predicted_token_lst)):
        for ans in answersets[i]:            
            if predicted_token_lst[i] == ans[0].lower():
                AM.add(i)
                print(predicted_token_lst[i],ans[0])
                
    print('AM: %f'%(len(AM)/len(predicted_token_lst)))  