import torch
import os
import tqdm
import json
import sys
import logging
from datetime import datetime
from transformers import BertTokenizer, BertForMaskedLM
from utils import extract_context
from utils import clean_paragraph
from utils import generate_bert_format_qas
from utils import generate_bert_format_context


"""
SpanBert model, with preprocessing
"""