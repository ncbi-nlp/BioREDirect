# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 13:27:17 2021

@author: laip2
"""

import os
import random
import glob
import xml.etree.ElementTree as ET
import re
import numpy as np
import sys
import optparse

sys.path.append('src/dataset_format_converter')
from document import PubtatorDocument, TextInstance
from annotation import AnnotationInfo, RelationInfo
from pathlib import Path
from transformers import BertTokenizer
import utils    

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from data_processor import BioREDDataset, CDRDataset

parser = optparse.OptionParser()
 
parser.add_option('--exp_option',                       action="store",
                     dest="exp_option",                 help="cdr; bc6pm", default="")

parser.add_option('--out_data_dir',                     action="store",
                     dest="out_data_dir",               help="output processed dir", default="")

parser.add_option('--in_bioc_xml_file',                 action="store",
                     dest="in_bioc_xml_file",           help="input bioc xml file for 'bioc_2_tsv'", default="")

parser.add_option('--in_bioc_xml_dir',                  action="store",
                     dest="in_bioc_xml_dir",            help="input bioc xml dir for 'bioc_2_tsv'", default="")

parser.add_option('--out_tsv_file',                     action="store",
                     dest="out_tsv_file",               help="output tsv file for 'bioc_2_tsv'", default="")

parser.add_option('--out_tsv_dir',                      action="store",
                     dest="out_tsv_dir",                help="output tsv dir for 'bioc_2_tsv'", default="")

parser.add_option('--sections',                         action="store",
                     dest="sections",                   help="full-text sections for processing", default="")

def add_annotations_2_text_instances(text_instances, annotations, title_shift = 0):
        
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
            #raise

    if title_shift != 0:
        for text_instance in text_instances:
            if text_instance.section != 'TITLE':
                text_instance.offset += title_shift

def load_bioc_into_documents(in_bioc_xml_file, 
                             normalized_ne_type_dict = {},
                             re_id_spliter_str = r'\;',
                             sections = ''):
    
    documents = []
    
    tree = ET.parse(in_bioc_xml_file)
    tree = tree.getroot()

    if sections == '':
        sections = {i.lower() for i in ['TITLE', 'ABSTRACT']}
    else:
        sections = {i.lower() for i in sections.split('|')}
    
    for doc_node in tree.findall('document'):
        
        pmid = doc_node.find('id').text
        #print(pmid)
        document = PubtatorDocument(pmid)

        annotations = []
        text_instances = []
        bioc_rel_id = 0
        refid_2_ids_dict = {}
        refid_2_ne_type_dict = {}
        id_2_ne_type_dict = {}
        relation_pairs = {} # for conversion
        
        # default sections are TITLE and ABSTRACT

        text_instances = []
        title_shift = 0
        for passage_node in doc_node.findall('passage'):
            #print(passage_node.find('text').text)
            if passage_node.find('text') == None:
                continue
            
            # below "section_type" only work whiles using full text
            # we only process TITLE and ABSTRACT
            #==============
            if passage_node.find('infon[@key="section_type"]') != None:
                if ('all' not in sections) and (passage_node.find('infon[@key="section_type"]').text.lower() not in sections):
                    continue
            #==============
            if passage_node.find('infon[@key="section_type"]') != None:
                # fix full text's title has no '.' at the end
                if passage_node.find('infon[@key="section_type"]').text == 'TITLE':
                    text_instance = TextInstance(passage_node.find('text').text + '.')
                    title_shift = 1 # '. ' will update in add_annotations_2_text_instances()
                else:
                    text_instance = TextInstance(passage_node.find('text').text)

                text_instance.section = passage_node.find('infon[@key="section_type"]').text
                if passage_node.find('infon[@key="section_type"]').text == 'ABSTRACT':
                    text_instance.offset = int(passage_node.find('offset').text.strip())
                else:
                    text_instance.offset = int(passage_node.find('offset').text)
            else:
                text_instance = TextInstance(passage_node.find('text').text)
                text_instance.section = passage_node.find('infon[@key="type"]').text
                text_instance.offset = int(passage_node.find('offset').text)
            #print(passage_offset)
            for annotation_node in passage_node.findall('annotation'):
                #annotation_node.attrib['id'] = str(10000 + int(annotation_node.attrib['id']))
                #print(annotation_node.find('text').text)
                bioc_annotation_id = annotation_node.attrib['id']
                
                ne_type = annotation_node.find('infon[@key="type"]').text
                identifier = None
                if annotation_node.find('infon[@key="identifier"]') != None:
                    identifier = annotation_node.find('infon[@key="identifier"]').text
                if annotation_node.find('infon[@key="annotator"]') != None:
                    annotator = annotation_node.find('infon[@key="annotator"]').text
                else:
                    annotator = ""
                if annotation_node.find('infon[@key="updated_at"]') != None:
                    updated_at = annotation_node.find('infon[@key="updated_at"]').text
                else:
                    updated_at = ''
                text = annotation_node.find('text').text
                offset = int(annotation_node.find('location').attrib['offset'])
                length = int(annotation_node.find('location').attrib['length'])
                if ne_type in normalized_ne_type_dict:
                    ne_type = normalized_ne_type_dict[ne_type]
                #print(ne_type)
                ann = AnnotationInfo(ne_type    = ne_type,
                                        text       = text,
                                        position   = offset,
                                        length     = length)
                ann.annotator = annotator
                ann.updated_at = updated_at
                if identifier != None:
                    ann.ids = set(re.split(re_id_spliter_str, identifier))
                else:
                    ann.ids = set([""])
                ann.bioc_id = annotation_node.attrib['id']
                if annotation_node.find('infon[@key="note"]') != None:
                    ann.note = annotation_node.find('infon[@key="note"]').text
                annotations.append(ann)
                refid_2_ids_dict[ann.bioc_id] = ann.ids
                refid_2_ne_type_dict[ann.bioc_id] = ann.ne_type
                for id in ann.ids:
                    id_2_ne_type_dict[id] = ann.ne_type    

            relations = []
            for relation_node in passage_node.findall('relation'):
                bioc_id = 'R' + str(bioc_rel_id)
                bioc_rel_id += 1
                rel = RelationInfo()
                rel.bioc_id = bioc_id
                rel.entities = []
                rel.type = relation_node.find('infon[@key="type"]').text
                rel.annotator = relation_node.find('infon[@key="annotator"]').text
                rel.updated_at = relation_node.find('infon[@key="updated_at"]').text
                for node in relation_node.findall('node'):
                    refid = node.attrib['refid']
                    role = node.attrib['role']
                    rel.entities.append([refid, role])
                if relation_node.find('infon[@key="note"]') != None:
                    rel.note = relation_node.find('infon[@key="note"]').text
                
                
                converters = set()
                
                for entity in rel.entities:
                    ne_type = refid_2_ne_type_dict[entity[0]]
                    if ne_type == 'GeneOrGeneProduct':
                        for id in refid_2_ids_dict[entity[0]]:
                            converters.add(id)
                
                for entity1 in rel.entities:
                    ne_type1 = refid_2_ne_type_dict[entity1[0]]
                    for id1 in refid_2_ids_dict[entity1[0]]:
                        for entity2 in rel.entities:
                            if entity2 == entity1:
                                continue
                            ne_type2 = refid_2_ne_type_dict[entity2[0]]
                            for id2 in refid_2_ids_dict[entity2[0]]:
                                if id1 == id2:
                                    continue
                                pair = (id1, id2) if id1 < id2 else (id2, id1)
                                
                                if pair not in relation_pairs:
                                    relation_pairs[pair] = rel.type
                                    
                #print(rel.type)
                relations.append(rel)
            text_instance.relations = relations
            text_instances.append(text_instance)
        
        relations = []
        for relation_node in doc_node.findall('relation'):
            
            bioc_id = 'R' + str(bioc_rel_id)
            bioc_rel_id += 1
            rel = RelationInfo()
            rel.bioc_id = bioc_id
            rel.type = relation_node.find('infon[@key="type"]').text
            if relation_node.find('infon[@key="annotator"]') != None:
                rel.annotator = relation_node.find('infon[@key="annotator"]').text
            if relation_node.find('infon[@key="updated_at"]') != None:
                rel.updated_at = relation_node.find('infon[@key="updated_at"]').text
            
            rel.entities = []
            for node in relation_node.findall('node'):
                refid = ''
                if node.find('infon[@key="type"]') != None:
                    refid = node.find('infon[@key="type"]').text
                if node.find('infon[@key="role"]') != None:
                    role = node.find('infon[@key="role"]').text
                if refid != '':
                    rel.entities.append([refid, role])
            if relation_node.find('infon[@key="note"]') != None:
                rel.note = relation_node.find('infon[@key="note"]').text
            
            relations.append(rel)
            
            
            for entity1 in rel.entities:
                ne_type1 = refid_2_ne_type_dict[entity1[0]]
                for id1 in refid_2_ids_dict[entity1[0]]:
                    for entity2 in rel.entities:
                        if entity2 == entity1:
                            continue
                        ne_type2 = refid_2_ne_type_dict[entity2[0]]
                        for id2 in refid_2_ids_dict[entity2[0]]:
                            if id1 == id2:
                                continue
                            pair = (id1, id2) if id1 < id2 else (id2, id1)
                            if pair not in relation_pairs:
                                relation_pairs[pair] = rel.type
                    
            # below for pubtator3 relations
             # below for biored.zip's bioc xml file
            pubtator3_id1 = ''
            pubtator3_id2 = ''
            if relation_node.find('infon[@key="role1"]') != None:
                pubtator3_ne_type1, pubtator3_id1 = relation_node.find('infon[@key="role1"]').text.split('|', 1)
                if pubtator3_ne_type1 in normalized_ne_type_dict:
                    pubtator3_ne_type1 = normalized_ne_type_dict[pubtator3_ne_type1]
            if relation_node.find('infon[@key="role2"]') != None:                
                pubtator3_ne_type2, pubtator3_id2 = relation_node.find('infon[@key="role2"]').text.split('|', 1)
                if pubtator3_ne_type2 in normalized_ne_type_dict:
                    pubtator3_ne_type2 = normalized_ne_type_dict[pubtator3_ne_type2]
            
            if pubtator3_id1 != '':
                # pubtator3_id is ordered
                for _pubtator3_id1 in pubtator3_id1.split(';'):
                    for _pubtator3_id2 in pubtator3_id2.split(';'):
                        pair = (_pubtator3_id1, _pubtator3_id2)
                        if pair not in relation_pairs:
                            relation_pairs[pair] = rel.type
            
        document.relation_pairs = relation_pairs
        document.relations = relations
        add_annotations_2_text_instances(text_instances, annotations, title_shift)
        document.text_instances = text_instances
        documents.append(document)
    return documents
        
def convert_bioc_to_tsv_file(
        in_bioc_xml_file,
        out_tsv_file,
        tokenizer,
        re_id_spliter_str       = r'\,',
        normalized_ne_type_dict = {},
        sections                = '',
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
            
    all_documents = load_bioc_into_documents(
            in_bioc_xml_file        = in_bioc_xml_file, 
            re_id_spliter_str       = re_id_spliter_str,
            normalized_ne_type_dict = normalized_ne_type_dict,
            sections                = sections)
        
    utils.split_documents(all_documents, tokenizer)

    utils.dump_documents_2_bioredirect_format(
        all_documents, 
        out_tsv_file,
        considered_ne_pairs,
        is_biored = task == 'biored', 
        tokenizer = tokenizer)

def gen_bioc_2_tsv_dataset(
        in_bioc_xml_file,
        in_bioc_xml_dir,
        out_tsv_file,
        out_tsv_dir,
        re_id_spliter_str,
        normalized_ne_type_dict,
        tokenizer,
        task):
    
    if in_bioc_xml_dir != '' and out_tsv_dir != '':

        if not os.path.exists(out_tsv_dir):
            os.makedirs(out_tsv_dir)
          
        for in_bioc_xml_file in glob.glob(in_bioc_xml_dir + '/*.txt') + glob.glob(in_bioc_xml_dir + '/*.xml'):
            
            file_name = Path(in_bioc_xml_file).stem
            out_tsv_file = out_tsv_dir + '/' + file_name + '.tsv'
            
            convert_bioc_to_tsv_file(
                in_bioc_xml_file  = in_bioc_xml_file,
                out_tsv_file      = out_tsv_file,
                re_id_spliter_str = re_id_spliter_str,
                normalized_ne_type_dict = normalized_ne_type_dict,
                tokenizer         = tokenizer,
                task              = task)
    else:
        
        os.makedirs(os.path.dirname(out_tsv_file), exist_ok=True)
        
        convert_bioc_to_tsv_file(
            in_bioc_xml_file  = in_bioc_xml_file,
            out_tsv_file      = out_tsv_file,
            re_id_spliter_str = re_id_spliter_str,
            normalized_ne_type_dict = normalized_ne_type_dict,
            tokenizer         = tokenizer,
            task              = task)

    
def init_tokenizer(options):

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
    
    random.seed(1234)
    options, args      = parser.parse_args()
    exp_option         = options.exp_option
    
    normalized_ne_type_dict = {
            "Chemical": "ChemicalEntity",
            "Disease": "DiseaseOrPhenotypicFeature",
            "Gene": "GeneOrGeneProduct",
            "DNAMutation": "GeneOrGeneProduct",
            "ProteinMutation": "GeneOrGeneProduct",
            "Mutation": "GeneOrGeneProduct",
            "SNP": "GeneOrGeneProduct",
            "Protein": "GeneOrGeneProduct",
            "Variant": "GeneOrGeneProduct",
            "SequenceVariant": "GeneOrGeneProduct",
            }
    

    tokenizer = init_tokenizer(options)

    if options.task == 'biored':
        re_id_spliter_str = r'[\,\;]'

        gen_bioc_2_tsv_dataset(
            in_bioc_xml_file        = options.in_bioc_xml_file,
            in_bioc_xml_dir         = options.in_bioc_xml_dir,
            out_tsv_file            = options.out_tsv_file,
            out_tsv_dir             = options.out_tsv_dir,
            re_id_spliter_str       = re_id_spliter_str,
            normalized_ne_type_dict = normalized_ne_type_dict,
            tokenizer               = tokenizer,
            task                    = options.task,
            sections                = options.sections)
            