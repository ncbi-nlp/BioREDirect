# -*- coding: utf-8 -*-

import glob
import os
from annotation import AnnotationInfo
import re
from pathlib import Path

import optparse

mutation_type_set = set(["DNAMutation", "ProteinMutation", "Mutation", "SNP", "SequenceVariant"])

__legal_pair_2_relation_types = {
            ("ChemicalEntity", "ChemicalEntity"): set(["Association", "Comparison", "Conversion", "Cotreatment", "Drug_Interaction", "Negative_Correlation", "Positive_Correlation"]),
            ("ChemicalEntity", "DiseaseOrPhenotypicFeature"): set(["Association", "Negative_Correlation", "Positive_Correlation"]),
            ("ChemicalEntity", "GeneOrGeneProduct"): set(["Association", "Bind", "Cotreatment", "Negative_Correlation", "Positive_Correlation"]),
            ("ChemicalEntity", "SequenceVariant"): set(["Association", "Negative_Correlation", "Positive_Correlation"]),
            ("DiseaseOrPhenotypicFeature", "GeneOrGeneProduct"): set(["Association", "Negative_Correlation", "Positive_Correlation"]),
            ("DiseaseOrPhenotypicFeature", "SequenceVariant"): set(["Association", "Negative_Correlation", "Positive_Correlation"]),
            ("GeneOrGeneProduct", "GeneOrGeneProduct"): set(["Association", "Bind", "Negative_Correlation", "Positive_Correlation"]),
            ("SequenceVariant", "SequenceVariant"): set(["Association"]),
            ("Chemical", "Chemical"): set(["Association", "Comparison", "Conversion", "Cotreatment", "Drug_Interaction", "Negative_Correlation", "Positive_Correlation"]),
            ("Chemical", "Disease"): set(["Association", "Negative_Correlation", "Positive_Correlation"]),
            ("Chemical", "Gene"): set(["Association", "Bind", "Cotreatment", "Negative_Correlation", "Positive_Correlation"]),
            ("Disease", "Gene"): set(["Association", "Negative_Correlation", "Positive_Correlation"]),
            ("Gene", "Gene"): set(["Association", "Bind", "Negative_Correlation", "Positive_Correlation"]),}

__legal_variant_relation_types = set(["Association", "Negative_Correlation", "Positive_Correlation"])

def dump_documents_2_pubtator3(
    in_data_file,
    out_data_file,
    pmid_2_rel_pair_dict,
    re_id_spliter_str = r'\;'):    
    
    id2ne_type_dict = {}
    id2index = {}
    
    out_str = ''
    with open(in_data_file, 'r', encoding='utf8') as pub_reader:
        
        pmid = ''
        
        annotations = []
        relation_pairs = {}
        id2ne_type = {}
        mutation_2_normalized_id_dict = {}
        mutation_2_gene_id_dict = {}
        
        for line in pub_reader:
            line = line.rstrip()
            _tks = line.split('\t')
            
            if line == '':
                
                if pmid in pmid_2_rel_pair_dict:
                    for rel in pmid_2_rel_pair_dict[pmid]:
                        
                        id1 = rel.id1
                        id2 = rel.id2

                        ne_type1 = id2ne_type[id1]
                        ne_type2 = id2ne_type[id2]
                        
                        if (id2ne_type[id1] == 'DiseaseOrPhenotypicFeature' or id2ne_type[id1] == 'Disease') and (id2ne_type[id2] in mutation_type_set) and (id2 in mutation_2_gene_id_dict):
                            relation_pairs[(id1, mutation_2_gene_id_dict[id2])] = ('Association', rel.rel_score)
                        
                        if id1 in mutation_2_normalized_id_dict:
                            id1 = mutation_2_normalized_id_dict[id1]
                        if id2 in mutation_2_normalized_id_dict:
                            id2 = mutation_2_normalized_id_dict[id2]
                        
                        pair = (ne_type1, ne_type2)
                            
                        if pair in __legal_pair_2_relation_types:
                            if rel.rel_type not in __legal_pair_2_relation_types[pair]:
                                rel.rel_type = 'Association'
                        elif (id1 in mutation_type_set) and (id2 not in mutation_type_set):
                            if rel.rel_type not in __legal_variant_relation_types:
                                rel.rel_type = 'Association'
                        elif (id2 in mutation_type_set) and (id1 not in mutation_type_set):
                            if rel.rel_type not in __legal_variant_relation_types:
                                rel.rel_type = 'Association'
                        elif (id2 in mutation_type_set) and (id1 in mutation_type_set):
                            rel.rel_type = 'Association'
                            
                        out_str += pmid + \
                                  '\t' + rel.rel_type + '|' + str(rel.rel_score) + \
                                  '\t' + id1 + \
                                  '\t' + id2 + \
                                  '\t' + rel.novelty_type + '|' + str(rel.novelty_score) + '\n'
                        
                        if rel.direction_type == 'Left_to_Right':
                            out_str += pmid + \
                                    '\t' + id1 + \
                                    '\t' + id2 + \
                                    '\t' + 'Subject:' + id1 + '|' + str(rel.direction_score) + '\n'
                        elif rel.direction_type == 'Right_to_Left':
                            out_str += pmid + \
                                    '\t' + id1 + \
                                    '\t' + id2 + \
                                    '\t' + 'Subject:' + id2 + '|' + str(rel.direction_score) + '\n'
                        
                    pmid = ''        
                out_str += '\n'
                annotations = []
                relation_pairs = {}
                mutation_2_normalized_id_dict = {}
                mutation_2_gene_id_dict = {}
                id2ne_type = {}

            elif len(_tks) == 6:
                    out_str += line + '\n'
                    start = int(_tks[1])
                    end = int(_tks[2])
                    index = _tks[1] + '|' + _tks[2]
                    text = _tks[3]
                    ne_type = _tks[4]
                    ne_type = re.sub(r'\s*\(.*?\)\s*$', '', ne_type)
                                            
                    orig_ne_type = ne_type
                    _anno = AnnotationInfo(start, end-start, text, ne_type)
                    
                    #2234245	250	270	audiovisual toxicity	Disease	D014786|D006311
                    ids = [x.strip('*') for x in re.split(re_id_spliter_str, _tks[5])]
                    
                    for id in ids:
                        id2ne_type[id] = ne_type
                    if ne_type in mutation_type_set:
                        rs_number = ''
                        HGVS = ''
                        gene_id = ''
                        for identifier in ids:
                            if identifier.startswith('RS#:'):
                                rs_number = identifier
                            elif identifier.startswith('HGVS:'):
                                HGVS = identifier
                            elif identifier.startswith('CorrespondingGene:'):
                                gene_id = identifier
                        for identifier in ids:
                            if identifier in mutation_2_normalized_id_dict:
                                continue
                            mutation_2_normalized_id_dict[identifier] = ';'.join(ids)
                            #wei's format
                            if rs_number != '' and HGVS != '' and gene_id != '':
                                mutation_2_normalized_id_dict[identifier] = rs_number + ';' + HGVS + ';' + gene_id
                            elif rs_number != '' and HGVS != '':
                                mutation_2_normalized_id_dict[identifier] = rs_number + ';' + HGVS
                            elif rs_number != '' and gene_id != '':
                                mutation_2_normalized_id_dict[identifier] = rs_number + ';' + gene_id
                            elif rs_number != '':
                                mutation_2_normalized_id_dict[identifier] = rs_number
                            elif HGVS != '' and gene_id != '':
                                mutation_2_normalized_id_dict[identifier] = HGVS + ';' + gene_id
                            elif HGVS != '':
                                mutation_2_normalized_id_dict[identifier] = HGVS
                            if gene_id != '':
                                mutation_2_gene_id_dict[identifier] = gene_id.split(':',1)[1]                         
                    
                    _anno.orig_ne_type = orig_ne_type
                    _anno.ids = set(ids)
                    annotations.append(_anno)
            else:
                tks = line.split('|')
                if len(tks) > 1 and (tks[1] == 't' or tks[1] == 'a'):
                    #2234245	250	270	audiovisual toxicity	Disease	D014786|D006311
                    pmid = tks[0]

                    out_str += line + '\n'
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
                        out_str += line + '\n'
                        #id = re.sub('\s*\(.*?\)\s*$', '', tks[5])
                        ne_type = tks[4]
                        
                        id2ne_type_dict[id] = ne_type
                        
                        id2index[id] = index
        if pmid != '' and pmid in pmid_2_rel_pair_dict:
            for rel in pmid_2_rel_pair_dict[pmid]:
                
                out_str += pmid + \
                            '\t' + rel.rel_type + \
                            '\t' + rel.id1 + \
                            '\t' + rel.id2 + \
                            '\t' + rel.rel_score + '\n'
    
    if os.path.exists(out_data_file):
        return
    with open(out_data_file, 'w', encoding='utf8') as pred_writer:
        pred_writer.write(out_str)
                