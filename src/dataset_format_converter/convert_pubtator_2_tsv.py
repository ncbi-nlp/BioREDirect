# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 13:27:17 2021

@author: laip2
"""

from document import PubtatorDocument, TextInstance
from annotation import AnnotationInfo
import os
import random
import glob
from pathlib import Path

import re

import sys
import utils
      
import optparse
from transformers import BertTokenizer

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from data_processor import BioREDDataset, CDRDataset

parser = optparse.OptionParser()

parser.add_option('--in_pubtator_file',                 action="store",
                     dest="in_pubtator_file",           help="input pubtator file for 'pubtator_2_tsv'", default="")

parser.add_option('--in_pubtator_dir',                  action="store",
                     dest="in_pubtator_dir",            help="input pubtator dir for 'pubtator_2_tsv'", default="")

parser.add_option('--out_tsv_file',                     action="store",
                     dest="out_tsv_file",               help="output tsv file for 'pubtator_2_tsv'", default="")

parser.add_option('--out_tsv_dir',                      action="store",
                     dest="out_tsv_dir",                help="output tsv dir for 'pubtator_2_tsv'", default="")

parser.add_option('--in_bert_model',                    action="store",
                     dest="in_bert_model",              help="input bert model", default="")

parser.add_option('--task',                             action="store",
                     dest="task",                       help="task name", default="biored")

def add_annotations_2_text_instances(text_instances, annotations):
    offset = 0
    for text_instance in text_instances:
        text_instance.offset = offset
        offset += len(text_instance.text) + 1
        
    for annotation in annotations:
        can_be_mapped_to_text_instance = False                
        for i, text_instance in enumerate(text_instances):
            if text_instance.offset <= annotation.position and annotation.position + annotation.length <= text_instance.offset + len(text_instance.text):
                
                annotation.position = annotation.position - text_instance.offset
                text_instance.annotations.append(annotation)
                can_be_mapped_to_text_instance = True
                break
        if not can_be_mapped_to_text_instance:
            print(annotation.text)
            print(annotation.position)
            print(annotation.length)
            print(annotation, 'cannot be mapped to original text')
            raise
    
def load_pubtator_into_documents(in_pubtator_file, 
                                 re_id_spliter_str           = r'\,',
                                 normalized_ne_type_dict     = {},
                                 use_novelty_label           = False):
    
    documents = []
    
    with open(in_pubtator_file, 'r', encoding='utf8') as pub_reader:
        
        pmid = ''
        
        document = None
        
        annotations = []
        text_instances = []
        relation_pairs = {}
        index2normalized_id = {}
        id2index = {}
        
        for line in pub_reader:
            line = line.rstrip()
            
            if line == '':
                
                document = PubtatorDocument(pmid)
                #print(pmid)
                add_annotations_2_text_instances(text_instances, annotations)
                document.text_instances = text_instances
                document.relation_pairs = relation_pairs
                documents.append(document)
                
                annotations = []
                text_instances = []
                relation_pairs = {}
                id2index = {}
                index2normalized_id = {}
                continue
            
            tks = line.split('|')
            
            
            if len(tks) > 1 and (tks[1] == 't' or tks[1] == 'a'):
                #2234245	250	270	audiovisual toxicity	Disease	D014786|D006311
                pmid = tks[0]
                x = TextInstance(tks[2])
                text_instances.append(x)
            else:
                _tks = line.split('\t')
                if len(_tks) == 6:
                    start = int(_tks[1])
                    end = int(_tks[2])
                    index = _tks[1] + '|' + _tks[2]
                    text = _tks[3]
                    ne_type = _tks[4]
                    ne_type = re.sub('\s*\(.*?\)\s*$', '', ne_type)
                    orig_ne_type = ne_type
                    if ne_type in normalized_ne_type_dict:
                        ne_type = normalized_ne_type_dict[ne_type]
                    
                    _anno = AnnotationInfo(start, end-start, text, ne_type)
                    
                    #2234245	250	270	audiovisual toxicity	Disease	D014786|D006311
                    ids = [x.strip('*') for x in re.split(re_id_spliter_str, _tks[5])]
                    
                    _anno.orig_ne_type = orig_ne_type
                    _anno.ids = set(ids)
                    annotations.append(_anno)
                elif len(_tks) == 4 or len(_tks) == 5:
                    
                    if len(_tks) == 4 and _tks[-1].startswith('Subject:'):
                        id1 = _tks[1]
                        id2 = _tks[2]

                        if id1 == '-' or id2 == '-':
                            continue
                        
                        subj_id = _tks[3].split(':', 1)[1] # "Subject:D123456|0.1234" or "Subject:D123456" => "D123456|0.1234" 

                        if subj_id.rsplit('|', 1)[0] == id1: # "p|xxx|xxx|0.1234" or "p|xxx|xxx" to "p|xxx|xxx"
                            subj_id = id1
                        elif subj_id.rsplit('|', 1)[0] == id2:
                            subj_id = id2

                        if (id1, id2) in relation_pairs:
                            relation_pairs[(id1, id2)] = relation_pairs[(id1, id2)] + '\t' + subj_id
                        else:
                            relation_pairs[(id2, id1)] = relation_pairs[(id2, id1)] + '\t' + subj_id
                    else:
                        id1 = _tks[2]
                        id2 = _tks[3]

                        if id1 == '-' or id2 == '-':
                            continue
                        
                        rel_type = _tks[1].split('|')[0]
                        
                        relation_pairs[(id1, id2)] = rel_type
                        if use_novelty_label and len(_tks) == 5:
                            relation_pairs[(id1, id2)] = rel_type + '\t' + _tks[-1].split('|')[0]
                        
        if len(text_instances) != 0:
            document = PubtatorDocument(pmid)
            add_annotations_2_text_instances(text_instances, annotations)
            document.text_instances = text_instances
            document.relation_pairs = relation_pairs
            documents.append(document)
    
    return documents

def convert_pubtator_to_tsv_file(
        in_pubtator_file,
        out_tsv_file,
        tokenizer,
        re_id_spliter_str       = r'\,',
        normalized_ne_type_dict = {},
        task                    = 'biored'):
    
    considered_ne_pairs = set([
        ('ChemicalEntity', 'ChemicalEntity'),
        ('ChemicalEntity', 'DiseaseOrPhenotypicFeature'),
        ('ChemicalEntity', 'GeneOrGeneProduct'),
        ('ChemicalEntity', 'SequenceVariant'),
        ('DiseaseOrPhenotypicFeature', 'GeneOrGeneProduct'),
        ('DiseaseOrPhenotypicFeature', 'SequenceVariant'),
        ('GeneOrGeneProduct', 'GeneOrGeneProduct'),
        ('SequenceVariant', 'SequenceVariant'),
        ('Chemical', 'Disease')])
            
    all_documents = load_pubtator_into_documents(
            in_pubtator_file     = in_pubtator_file, 
            re_id_spliter_str    = re_id_spliter_str,
            normalized_ne_type_dict = normalized_ne_type_dict,
            use_novelty_label    = task == 'biored')
        
    utils.split_documents(all_documents, tokenizer)

    utils.dump_documents_2_bioredirect_format(
        all_documents, 
        out_tsv_file,
        considered_ne_pairs,
        is_biored = task == 'biored', 
        tokenizer = tokenizer)

def gen_pubtator_2_tsv_dataset(
        in_pubtator_file,
        in_pubtator_dir,
        out_tsv_file,
        out_tsv_dir,
        re_id_spliter_str,
        normalized_ne_type_dict,
        tokenizer,
        task):
    
    if in_pubtator_dir != '' and out_tsv_dir != '':

        if not os.path.exists(out_tsv_dir):
            os.makedirs(out_tsv_dir)
          
        for in_pubtator_file in glob.glob(in_pubtator_dir + '/*.txt'):
            
            file_name = Path(in_pubtator_file).stem
            out_tsv_file = out_tsv_dir + '/' + file_name + '.tsv'
            
            convert_pubtator_to_tsv_file(
                in_pubtator_file  = in_pubtator_file,
                out_tsv_file      = out_tsv_file,
                re_id_spliter_str = re_id_spliter_str,
                normalized_ne_type_dict = normalized_ne_type_dict,
                tokenizer         = tokenizer,
                task              = task)
    else:
        
        if os.path.dirname(out_tsv_file) != '':
            os.makedirs(os.path.dirname(out_tsv_file), exist_ok=True)
        
        convert_pubtator_to_tsv_file(
            in_pubtator_file  = in_pubtator_file,
            out_tsv_file      = out_tsv_file,
            re_id_spliter_str = re_id_spliter_str,
            normalized_ne_type_dict = normalized_ne_type_dict,
            tokenizer         = tokenizer,
            task              = task)
    
def init_tokenizer(in_bert_model):

    tokenizer = BertTokenizer.from_pretrained(options.in_bert_model)

    if options.task == 'biored':
        dataset_processor = BioREDDataset
    elif options.task == 'cdr':
        dataset_processor = CDRDataset

    # Get additional special tokens
    additional_special_tokens = dataset_processor.get_special_tokens()

    # Merge vocabularies
    merged_vocab = tokenizer.get_vocab()
    to_add_special_tokens = []
    for token in additional_special_tokens:
        if (token not in merged_vocab) and (token not in tokenizer.additional_special_tokens):
            to_add_special_tokens.append(token)
            #print(token)

    for rel_token in ['[REL]', '[DIR]', '[NOV]']:
        if (rel_token not in merged_vocab) and (rel_token not in tokenizer.additional_special_tokens):
            to_add_special_tokens.append(rel_token)

    tokenizer.add_special_tokens({'additional_special_tokens': tokenizer.additional_special_tokens + to_add_special_tokens})

    return tokenizer

if __name__ == '__main__':
    
    options, args = parser.parse_args()

    random.seed(1111)

    tokenizer = init_tokenizer(options.in_bert_model)
    normalized_ne_type_dict = {}

    if options.task == 'cdr':
        re_id_spliter_str = r'[\,\;\|]'
    else:
        re_id_spliter_str = r'[\,\;]'
        normalized_ne_type_dict = {'Chemical': 'ChemicalEntity', 
                                   'Disease': 'DiseaseOrPhenotypicFeature',
                                   'Gene': 'GeneOrGeneProduct',}

    gen_pubtator_2_tsv_dataset(
        in_pubtator_file        = options.in_pubtator_file,
        in_pubtator_dir         = options.in_pubtator_dir,
        out_tsv_file            = options.out_tsv_file,
        out_tsv_dir             = options.out_tsv_dir,
        re_id_spliter_str       = re_id_spliter_str,
        normalized_ne_type_dict = normalized_ne_type_dict,
        tokenizer               = tokenizer,
        task                    = options.task)
    