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
import utils
      
import optparse


parser = optparse.OptionParser()
 
parser.add_option('--in_pubtator_file',                 action="store",
                     dest="in_pubtator_file",           help="input pubtator file for 'pubtator_2_tsv'", default="")

parser.add_option('--in_pubtator_dir',                  action="store",
                     dest="in_pubtator_dir",            help="input pubtator dir for 'pubtator_2_tsv'", default="")

parser.add_option('--out_json_file',                     action="store",
                     dest="out_json_file",               help="output json file for 'pubtator_2_json'", default="")

parser.add_option('--out_json_dir',                      action="store",
                     dest="out_json_dir",                help="output json dir for 'pubtator_2_json'", default="")

parser.add_option('--in_bert_model',                    action="store",
                     dest="in_bert_model",              help="input bert model", default="")


parser.add_option('--exp_option',                       action="store",
                     dest="exp_option",                 help="gen_biored_instruction_json, gen_cdr_instruction_json, and gen_cdr_zs_instruction_jsonl etc", default="")

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
                        #print(pmid)
                        if (id1, id2) in relation_pairs:
                            relation_pairs[(id1, id2)] = relation_pairs[(id1, id2)] + '\t' + _tks[3].split(':')[1]
                        else:
                            relation_pairs[(id2, id1)] = relation_pairs[(id2, id1)] + '\t' + _tks[3].split(':')[1]
                    else:
                        id1 = _tks[2]
                        id2 = _tks[3]
                        rel_type = _tks[1]                        
                        relation_pairs[(id1, id2)] = rel_type
                        if use_novelty_label and len(_tks) == 5:
                            relation_pairs[(id1, id2)] = rel_type + '\t' + _tks[-1]
                    
        if len(text_instances) != 0:
            document = PubtatorDocument(pmid)
            add_annotations_2_text_instances(text_instances, annotations)
            document.text_instances = text_instances
            document.relation_pairs = relation_pairs
            documents.append(document)
    
    return documents
    
def convert_biored_to_zs_instruction_json_file(
        in_pubtator_file,
        out_json_file,
        re_id_spliter_str,
        use_novelty_label = True):
    
    considered_ne_pairs = set([
        ('ChemicalEntity', 'ChemicalEntity'),
        ('ChemicalEntity', 'DiseaseOrPhenotypicFeature'),
        ('ChemicalEntity', 'GeneOrGeneProduct'),
        ('ChemicalEntity', 'SequenceVariant'),
        ('DiseaseOrPhenotypicFeature', 'GeneOrGeneProduct'),
        ('DiseaseOrPhenotypicFeature', 'SequenceVariant'),
        ('GeneOrGeneProduct', 'GeneOrGeneProduct'),
        ('SequenceVariant', 'SequenceVariant')])
            
    all_documents = load_pubtator_into_documents(
            in_pubtator_file     = in_pubtator_file, 
            re_id_spliter_str    = re_id_spliter_str,
            use_novelty_label    = use_novelty_label)
    
    utils.split_documents(all_documents, tokenizer)

    utils.dump_documents_2_zs_instruction_format(
        all_documents, 
        out_json_file,
        considered_ne_pairs)
    
def convert_cdr_to_zs_instruction_json_file(
        in_pubtator_file,
        out_json_file,
        re_id_spliter_str):
    
    considered_ne_pairs = set([('Chemical', 'Disease')])
            
    all_documents = load_pubtator_into_documents(
            in_pubtator_file     = in_pubtator_file, 
            re_id_spliter_str    = re_id_spliter_str)
    
    utils.split_documents(all_documents, tokenizer)

    utils.dump_documents_2_zs_instruction_format(
        all_documents, 
        out_json_file,
        considered_ne_pairs)
    
def convert_biored_to_instruction_json_file(
        in_pubtator_file,
        out_json_file,
        re_id_spliter_str,
        use_novelty_label = True):
    
    considered_ne_pairs = set([
        ('ChemicalEntity', 'ChemicalEntity'),
        ('ChemicalEntity', 'DiseaseOrPhenotypicFeature'),
        ('ChemicalEntity', 'GeneOrGeneProduct'),
        ('ChemicalEntity', 'SequenceVariant'),
        ('DiseaseOrPhenotypicFeature', 'GeneOrGeneProduct'),
        ('DiseaseOrPhenotypicFeature', 'SequenceVariant'),
        ('GeneOrGeneProduct', 'GeneOrGeneProduct'),
        ('SequenceVariant', 'SequenceVariant')])
            
    all_documents = load_pubtator_into_documents(
            in_pubtator_file     = in_pubtator_file, 
            re_id_spliter_str    = re_id_spliter_str,
            use_novelty_label    = use_novelty_label)
    
    #utils.split_documents(all_documents, tokenizer)

    utils.dump_documents_2_instruction_format(
        all_documents, 
        out_json_file,
        considered_ne_pairs)
    
def convert_cdr_to_instruction_json_file(
        in_pubtator_file,
        out_json_file,
        re_id_spliter_str,
        use_novelty_label = True):
    
    considered_ne_pairs = set([
        ('Chemical', 'Disease')])
            
    all_documents = load_pubtator_into_documents(
            in_pubtator_file     = in_pubtator_file, 
            re_id_spliter_str    = re_id_spliter_str,
            use_novelty_label    = use_novelty_label)
    
    utils.split_documents(all_documents, tokenizer)

    utils.dump_documents_2_instruction_format(
        all_documents, 
        out_json_file,
        considered_ne_pairs)

def gen_biored_instruction_json_dataset(
        in_pubtator_file,
        in_pubtator_dir,
        out_json_file,
        out_json_dir,
        re_id_spliter_str):
    
    if in_pubtator_dir != '' and out_json_dir != '':

        if not os.path.exists(out_json_dir):
            os.makedirs(out_json_dir)
          
        for in_pubtator_file in glob.glob(in_pubtator_dir + '/*.txt'):
            
            file_name = Path(in_pubtator_file).stem
            out_json_file = out_json_dir + '/' + file_name + '.json'
            
            convert_biored_to_instruction_json_file(
                in_pubtator_file  = in_pubtator_file,
                out_json_file     = out_json_file,
                re_id_spliter_str = re_id_spliter_str)
    else:
        
        if os.path.dirname(out_json_file) != '':
            os.makedirs(os.path.dirname(out_json_file), exist_ok=True)
        
        convert_biored_to_instruction_json_file(
            in_pubtator_file  = in_pubtator_file,
            out_json_file     = out_json_file,
            re_id_spliter_str = re_id_spliter_str)
        
def gen_cdr_instruction_json_dataset(
        in_pubtator_file,
        in_pubtator_dir,
        out_json_file,
        out_json_dir,
        re_id_spliter_str):
    
    if in_pubtator_dir != '' and out_json_dir != '':

        if not os.path.exists(out_json_dir):
            os.makedirs(out_json_dir)
          
        for in_pubtator_file in glob.glob(in_pubtator_dir + '/*.txt'):
            
            file_name = Path(in_pubtator_file).stem
            out_json_file = out_json_dir + '/' + file_name + '.json'
            
            convert_cdr_to_instruction_json_file(
                in_pubtator_file  = in_pubtator_file,
                out_json_file     = out_json_file,
                re_id_spliter_str = re_id_spliter_str)
    else:
        
        os.makedirs(os.path.dirname(out_json_file), exist_ok=True)
        
        convert_cdr_to_instruction_json_file(
            in_pubtator_file  = in_pubtator_file,
            out_json_file     = out_json_file,
            re_id_spliter_str = re_id_spliter_str)

def gen_biored_zs_instruction_json_dataset(
        in_pubtator_file,
        in_pubtator_dir,
        out_json_file,
        out_json_dir,
        re_id_spliter_str,
        tokenizer):
    
    if in_pubtator_dir != '' and out_json_dir != '':

        if not os.path.exists(out_json_dir):
            os.makedirs(out_json_dir)
          
        for in_pubtator_file in glob.glob(in_pubtator_dir + '/*.txt'):
            
            file_name = Path(in_pubtator_file).stem
            out_json_file = out_json_dir + '/' + file_name + '.json'
            
            convert_biored_to_zs_instruction_json_file(
                in_pubtator_file  = in_pubtator_file,
                out_json_file     = out_json_file,
                re_id_spliter_str = re_id_spliter_str)
    else:
        
        os.makedirs(os.path.dirname(out_json_file), exist_ok=True)
        
        convert_biored_to_zs_instruction_json_file(
            in_pubtator_file  = in_pubtator_file,
            out_json_file     = out_json_file,
            re_id_spliter_str = re_id_spliter_str)
        
def gen_cdr_zs_instruction_json_dataset(
        in_pubtator_file,
        in_pubtator_dir,
        out_json_file,
        out_json_dir,
        re_id_spliter_str):
    
    if in_pubtator_dir != '' and out_json_dir != '':

        if not os.path.exists(out_json_dir):
            os.makedirs(out_json_dir)
          
        for in_pubtator_file in glob.glob(in_pubtator_dir + '/*.txt'):
            
            file_name = Path(in_pubtator_file).stem
            out_json_file = out_json_dir + '/' + file_name + '.json'
            
            convert_cdr_to_llama_zs_instruction_json_file(
                in_pubtator_file  = in_pubtator_file,
                out_json_file     = out_json_file,
                re_id_spliter_str = re_id_spliter_str)
    else:
        
        os.makedirs(os.path.dirname(out_json_file), exist_ok=True)
        
        convert_cdr_to_llama_zs_instruction_json_file(
            in_pubtator_file  = in_pubtator_file,
            out_json_file     = out_json_file,
            re_id_spliter_str = re_id_spliter_str)

if __name__ == '__main__':
    
    options, args      = parser.parse_args()
    
    random.seed(1111)

    # standard train and dev
    if options.exp_option == 'gen_biored_instruction_json':
        in_pubtator_file  = options.in_pubtator_file
        in_pubtator_dir   = options.in_pubtator_dir
        out_json_file     = options.out_json_file
        out_json_dir      = options.out_json_dir
        re_id_spliter_str = r'[\,\;]'

        gen_biored_instruction_json_dataset(
            in_pubtator_file  = in_pubtator_file,
            in_pubtator_dir   = in_pubtator_dir,
            out_json_file     = out_json_file,
            out_json_dir      = out_json_dir,
            re_id_spliter_str = re_id_spliter_str)

    elif options.exp_option == 'gen_cdr_instruction_json':
        in_pubtator_file  = options.in_pubtator_file
        in_pubtator_dir   = options.in_pubtator_dir
        out_json_file     = options.out_json_file
        out_json_dir      = options.out_json_dir
        re_id_spliter_str = r'[\,\;\|]'

        gen_cdr_instruction_json_dataset(
            in_pubtator_file  = in_pubtator_file,
            in_pubtator_dir   = in_pubtator_dir,
            out_json_file     = out_json_file,
            out_json_dir      = out_json_dir,
            re_id_spliter_str = re_id_spliter_str)

    elif options.exp_option == 'gen_biored_zs_instruction_jsonl':
        in_pubtator_file  = options.in_pubtator_file
        in_pubtator_dir   = options.in_pubtator_dir
        out_json_file     = options.out_json_file
        out_json_dir      = options.out_json_dir
        re_id_spliter_str = r'[\,\;]'

        gen_biored_zs_instruction_json_dataset(
            in_pubtator_file  = in_pubtator_file,
            in_pubtator_dir   = in_pubtator_dir,
            out_json_file     = out_json_file,
            out_json_dir      = out_json_dir,
            re_id_spliter_str = re_id_spliter_str)
    
    elif options.exp_option == 'gen_cdr_zs_instruction_jsonl':
        in_pubtator_file = options.in_pubtator_file
        in_pubtator_dir  = options.in_pubtator_dir
        out_json_file    = options.out_json_file
        out_json_dir     = options.out_json_dir
        re_id_spliter_str= r'[\,\;\|]'

        gen_cdr_zs_instruction_json_dataset(
            in_pubtator_file  = in_pubtator_file,
            in_pubtator_dir   = in_pubtator_dir,
            out_json_file     = out_json_file,
            out_json_dir      = out_json_dir,
            re_id_spliter_str = re_id_spliter_str)

        
