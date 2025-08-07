import torch
import numpy as np
import os
import argparse

import logging
from tqdm import tqdm
from transformers import set_seed
from transformers import BertTokenizer
from data_processor import BioREDDataset, CDRDataset
from models import BioREDirect
from torch.utils.data import DataLoader
from evaluation import evaluate_biored_f1_score, evaluate_cdr_f1_score
from torch import nn
import torch.nn.functional as F

from run_exp import load_tokenizer, custom_collate_fn
from evaluation import convert_to_biored_label
import pandas as pd

parser = argparse.ArgumentParser(description='Test the model on the test set')
parser.add_argument('--in_test_pubtator_file', type=str, help='The input test pubtator file', required=True)
parser.add_argument('--in_test_tsv_file', type=str, help='The input test tsv file', required=True)
parser.add_argument('--in_pred_tsv_file', type=str, help='The input pred tsv file', required=True)
parser.add_argument('--out_pred_pubtator_file', type=str, help='The output pred pubtator file', required=True)
parser.add_argument('--to_pubtator3', action='store_true', help='Convert the predicate to pubtator3 format')

_pair_rel_2_pubtator3_predicate_dict = {
    
}

class RelInfo:

    def __init__(self, 
                 id1, 
                 id2, 
                 rel_label, 
                 rel_score,
                 nov_label,
                 nov_score,
                 dir_label,
                 dir_score):
        
        self.id1 = id1
        self.id2 = id2
        self.rel_label = rel_label
        self.rel_score = rel_score
        self.nov_label = nov_label
        self.nov_score = nov_score
        self.dir_label = dir_label
        self.dir_score = dir_score

def add_relation_pairs_dict(
        in_test_tsv_file, 
        in_pred_tsv_file,
        pmid_2_rel_pairs_dict,
        to_pubtator3):
    
    #print('in_gold_tsv_file', in_gold_tsv_file)
    #print('in_pred_tsv_file', in_pred_tsv_file)

    testdf = pd.read_csv(in_test_tsv_file, sep="\t", header=None)
    try:
        # contains header
        preddf = pd.read_csv(in_pred_tsv_file, sep="\t", header=0)
    except:
        return
    
    index_list            = testdf.iloc[:,0]
    id1_list              = testdf.iloc[:,3]
    id2_list              = testdf.iloc[:,4]

    preds = [preddf.iloc[i].tolist() for i in preddf.index]
    relation_labels = ['None',
                       'Association',
                       'Bind',
                       'Comparison',
                       'Conversion',
                       'Cotreatment',
                       'Drug_Interaction',
                       'Negative_Correlation',
                       'Positive_Correlation']
    novelty_labels = ['None', 'Novel', 'No']
    direction_labels = ['None', 'Left_to_Right', 'Right_to_Left', 'No_Direct']
    
    _counter = 0
    for index, id1, id2, pred in zip(index_list, id1_list, id2_list, preds):

        rel_scores = []
        for i in range(0, len(relation_labels)):
            rel_scores.append(float(pred[i]))
            
        nov_scores = []
        for i in range(len(relation_labels), len(relation_labels) + len(novelty_labels)):
            nov_scores.append(float(pred[i]))

        dir_scores = []
        for i in range(len(relation_labels) + len(novelty_labels), len(relation_labels) + len(novelty_labels) + len(direction_labels)):
            dir_scores.append(float(pred[i]))

        max_rel_score = 0
        rel_label = ''

        for score, label in zip(rel_scores, relation_labels):
            if score >= 0.5 and score >= max_rel_score and (label != 'Association' and label != 'None'):
                rel_label = label
                max_rel_score = score
            
        if rel_label == '':
            for score, label in zip(rel_scores, relation_labels):
                if score >= max_rel_score:
                    rel_label = label
                    max_rel_score = score

        max_rel_score = 0
        nov_label = ''
        max_pos_nov_score = 0
        max_pos_nov_label = ''
        for score, label in zip(nov_scores, novelty_labels):
            if score >= max_rel_score:
                nov_label = label
                max_rel_score = score
            if label != 'None' and score >= max_pos_nov_score:
                max_pos_nov_score = score
                max_pos_nov_label = label
                
        dir_label = ''
        max_pos_dir_score = 0
        for score, label in zip(dir_scores, direction_labels):
            if label != 'None' and score >= max_pos_dir_score:
                max_pos_dir_score = score
                dir_label = label
        
        if rel_label != 'None':
            if nov_label == 'None':
                nov_label = max_pos_nov_label
        
        sindex = str(index)
        if sindex not in pmid_2_rel_pairs_dict:
            pmid_2_rel_pairs_dict[sindex] = set()
            #print(sindex)
        if rel_label != 'None':
            pmid_2_rel_pairs_dict[sindex].add(RelInfo(id1, id2, 
                                                    rel_label, max_rel_score, 
                                                    nov_label, max_pos_nov_score, 
                                                    dir_label, max_pos_dir_score))
        _counter += 1
            
def dump_pred_2_pubtator_file(in_test_pubtator_file, 
                              in_test_tsv_file,
                              in_pred_tsv_file,
                              out_pred_pubtator_file,
                              to_pubtator3):
    
    pmid_2_rel_pairs_dict = {}
    add_relation_pairs_dict(
            in_test_tsv_file,
            in_pred_tsv_file,
            pmid_2_rel_pairs_dict,
            to_pubtator3)
    
    
    pred_writer = open(out_pred_pubtator_file, 'w', encoding='utf8')
        
    id2ne_type_dict = {}
    id2index = {}
            
    with open(in_test_pubtator_file, 'r', encoding='utf8') as pub_reader:
        
        pmid = ''
        
        for line in pub_reader:
            line = line.rstrip()
            
            if line == '':
                
                if pmid in pmid_2_rel_pairs_dict:
                    for rel in pmid_2_rel_pairs_dict[pmid]:
                        
                        id1 = rel.id1
                        id2 = rel.id2

                        pred_writer.write(pmid + 
                                  '\t' + rel.rel_label + '|' + str(rel.rel_score) + 
                                  '\t' + id1 + 
                                  '\t' + id2 + 
                                  '\t' + rel.nov_label + '|' + str(rel.nov_score) + '\n')
                        
                        if rel.dir_label == 'Left_to_Right':
                            pred_writer.write(pmid + 
                                    '\t' + id1 + 
                                    '\t' + id2 + 
                                    '\t' + 'Subject:' + id1 + '|' + str(rel.dir_score) + '\n')
                        elif rel.dir_label == 'Right_to_Left':
                            pred_writer.write(pmid + 
                                    '\t' + id1 + 
                                    '\t' + id2 + 
                                    '\t' + 'Subject:' + id2 + '|' + str(rel.dir_score) + '\n')
                        
                    pmid = ''        
                pred_writer.write('\n')
                id2ne_type_dict = {}
                id2index = {}
                        
            else:
                tks = line.split('|')
                if len(tks) > 1 and (tks[1] == 't' or tks[1] == 'a'):
                    #2234245	250	270	audiovisual toxicity	Disease	D014786|D006311
                    pmid = tks[0]
                    
                    pred_writer.write(line + '\n')
                else:
                    tks = line.split('\t')
                    pmid = tks[0]
                    if len(tks) == 6:
                        #print(line)
                        
                        id = tks[5]
                        start = tks[1]
                        end = tks[2]
                        index = start + '|' + end
                        
                        line = '\t'.join(tks)
                        pred_writer.write(line + '\n')
                        #id = re.sub('\s*\(.*?\)\s*$', '', tks[5])
                        ne_type = tks[4]
                        
                        id2ne_type_dict[id] = ne_type
                        
                        id2index[id] = index
        if pmid != '' and pmid in pmid_2_rel_pairs_dict:
            for rel in pmid_2_rel_pairs_dict[pmid]:
                
                pred_writer.write(pmid + 
                                  '\t' + rel.rel_type + 
                                  '\t' + rel.id1 + 
                                  '\t' + rel.id2 + 
                                  '\t' + rel.score + '\n')
                
    pred_writer.close()
    


if __name__ == '__main__':

    args = parser.parse_args() 

    in_test_pubtator_file    = args.in_test_pubtator_file
    in_test_tsv_file         = args.in_test_tsv_file
    in_pred_tsv_file         = args.in_pred_tsv_file
    out_pred_pubtator_file   = args.out_pred_pubtator_file
    to_pubtator3             = args.to_pubtator3

    dump_pred_2_pubtator_file(in_test_pubtator_file  = in_test_pubtator_file, 
                              in_test_tsv_file       = in_test_tsv_file,
                              in_pred_tsv_file       = in_pred_tsv_file,
                              out_pred_pubtator_file = out_pred_pubtator_file,
                              to_pubtator3           = to_pubtator3)