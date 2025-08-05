from document import PubtatorDocument, TextInstance
from annotation import AnnotationInfo
import datetime
from collections import defaultdict
import xml.etree.ElementTree as ET
import os
import numpy as np
import copy

class PubTator3Postprocessor:

    
    _legal_pair_2_relation_types = {
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

    _legal_variant_relation_types = set(["Association", "Negative_Correlation", "Positive_Correlation"])
    _mutation_type_set = set(["DNAMutation", "ProteinMutation", "Mutation", "SNP"])

    @staticmethod
    def _sub_element_with_text(
        parent, 
        tag, 
        attrib_key,
        attrib_value,
        text):
        attrib       = {attrib_key: attrib_value}
        element      = parent.makeelement(tag, attrib)
        parent.append(element)
        element.text = text.replace('&', '&#38;').replace('<', '&lt;').replace('>', '&gt;').replace('\'', '&#39;').replace('\"', '&#34;')
        return element


    @staticmethod
    def _sub_element(
        parent, 
        id, 
        rel_type,
        score,
        role1,
        role2):
        attrib = {'id': id}
        element = parent.makeelement('relation', attrib)
        PubTator3Postprocessor._sub_element_with_text(
            parent       = element, 
            tag          = 'infon',
            attrib_key   = 'key',
            attrib_value = 'type',
            text         = rel_type)
        PubTator3Postprocessor._sub_element_with_text(
            parent       = element, 
            tag          = 'infon',
            attrib_key   = 'key',
            attrib_value = 'score',
            text         = score)
        PubTator3Postprocessor._sub_element_with_text(
            parent       = element, 
            tag          = 'infon',
            attrib_key   = 'key',
            attrib_value = 'role1',
            text         = role1)
        PubTator3Postprocessor._sub_element_with_text(
            parent       = element, 
            tag          = 'infon',
            attrib_key   = 'key',
            attrib_value = 'role2',
            text         = role2)
        parent.append(element)

    @staticmethod
    def get_mutation_2_normalized_id_dict(doc_node):

        mutation_2_normalized_id_dict = {}
        mutation_2_gene_id_dict = {}
        for passage_node in doc_node.findall('passage'):

            # below "section_type" only work whiles using full text
            # we only process TITLE and ABSTRACT
            #==============
            if passage_node.find('infon[@key="section_type"]') != None:
                if passage_node.find('infon[@key="section_type"]').text != 'TITLE' and passage_node.find('infon[@key="section_type"]').text != 'ABSTRACT':
                    continue
            #==============

            for annotation_node in passage_node.findall('annotation'):
                if annotation_node.find('infon[@key="identifier"]') == None:
                    continue
                ne_type = annotation_node.find('infon[@key="type"]').text
                if annotation_node.find('infon[@key="identifier"]').text != None:
                    identifiers = list(annotation_node.find('infon[@key="identifier"]').text.split(';'))
                else:
                    identifiers = []
                
                if ne_type in PubTator3Postprocessor._mutation_type_set:
                    rs_number = ''
                    HGVS = ''
                    gene_id = ''
                    for identifier in identifiers:
                        if identifier.startswith('RS#:'):
                            rs_number = identifier
                        elif identifier.startswith('HGVS:'):
                            HGVS = identifier
                        elif identifier.startswith('CorrespondingGene:'):
                            gene_id = identifier
                    for identifier in identifiers:
                        if identifier in mutation_2_normalized_id_dict:
                            continue
                        mutation_2_normalized_id_dict[identifier] = ';'.join(identifiers)
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

        return mutation_2_normalized_id_dict, mutation_2_gene_id_dict

    # class method
    @staticmethod
    def add_gene_disease_via_mutation(
        doc_node, 
        mutation_2_gene_id_dict):
        counter = 1000
        for relation_node in doc_node.findall('relation'):
            ne_type1, id1  = relation_node.find('infon[@key="role1"]').text.split('|', 1)
            ne_type2, id2  = relation_node.find('infon[@key="role2"]').text.split('|', 1)
            rel_type       = relation_node.find('infon[@key="type"]').text
            score          = relation_node.find('infon[@key="score"]').text
            if (ne_type1 in PubTator3Postprocessor._mutation_type_set) and ne_type2 == 'Disease':
                if id1 in mutation_2_gene_id_dict:
                    gene_id    = mutation_2_gene_id_dict[id1]
                    PubTator3Postprocessor._sub_element(
                        parent = doc_node,
                        id     = 'R' + str(counter),
                        rel_type = 'Association',
                        score  = score,
                        role1  = 'Disease|' + id2,
                        role2  = 'Gene|' + gene_id)
            elif (ne_type2 in PubTator3Postprocessor._mutation_type_set) and ne_type1 == 'Disease':
                if id2 in mutation_2_gene_id_dict:
                    gene_id    = mutation_2_gene_id_dict[id2]
                    PubTator3Postprocessor._sub_element(
                        parent = doc_node,
                        id     = 'R' + str(counter),
                        rel_type = 'Association',
                        score  = score,
                        role1  = 'Disease|' + id1,
                        role2  = 'Gene|' + gene_id)
            counter += 1

    @staticmethod
    def remove_redundant_mutation_relation(
        doc_node,
        mutation_2_normalized_id_dict):
        
        for relation_node in doc_node.findall('relation'):
            ne_type1, id1  = relation_node.find('infon[@key="role1"]').text.split('|', 1)
            ne_type2, id2  = relation_node.find('infon[@key="role2"]').text.split('|', 1)
            
            if (ne_type1 in PubTator3Postprocessor._mutation_type_set) and (id1 not in mutation_2_normalized_id_dict):
                doc_node.remove(relation_node)
            elif (ne_type2 in PubTator3Postprocessor._mutation_type_set) and (id2 not in mutation_2_normalized_id_dict):
                doc_node.remove(relation_node)
            elif id1 == id2:
                doc_node.remove(relation_node)

    @staticmethod
    def normalize_mutation_id(
        doc_node, 
        mutation_2_normalized_id_dict):
        for relation_node in doc_node.findall('relation'):
            ne_type1, id1  = relation_node.find('infon[@key="role1"]').text.split('|', 1)
            ne_type2, id2  = relation_node.find('infon[@key="role2"]').text.split('|', 1)
            if (ne_type1 in PubTator3Postprocessor._mutation_type_set) and (id1 in mutation_2_normalized_id_dict):
                #print('before', relation_node.find('infon[@key="role1"]').text)
                relation_node.find('infon[@key="role1"]').text = ne_type1 + '|' + mutation_2_normalized_id_dict[id1]
                #print('after', relation_node.find('infon[@key="role1"]').text)
            if (ne_type2 in PubTator3Postprocessor._mutation_type_set) and (id2 in mutation_2_normalized_id_dict):
                relation_node.find('infon[@key="role2"]').text = ne_type2 + '|' + mutation_2_normalized_id_dict[id2]

    @staticmethod
    def update_relation_order(
        doc_node):
    
        for relation_node in doc_node.findall('relation'):
            info1 = relation_node.find('infon[@key="role1"]')
            info2 = relation_node.find('infon[@key="role2"]')
            ne_type1, id1  = info1.text.split('|', 1)
            ne_type2, id2  = info2.text.split('|', 1)
            
            if ne_type1 in PubTator3Postprocessor._mutation_type_set:
                ne_type1 = 'Variant'
            if ne_type2 in PubTator3Postprocessor._mutation_type_set:
                ne_type2 = 'Variant'
            
            if ne_type2 < ne_type1:
                info1.attrib["key"] = 'role2'
                info2.attrib["key"] = 'role1'
                for _node in relation_node.findall('node'):
                    _roles = _node.attrib["role"].split(',')
                    _node.attrib["role"] = _roles[1] + ',' + _roles[0]
            if ne_type1 == ne_type2 and id2 < id1:
                info1.attrib["key"] = 'role2'
                info2.attrib["key"] = 'role1'
                for _node in relation_node.findall('node'):
                    _roles = _node.attrib["role"].split(',')
                    _node.attrib["role"] = _roles[1] + ',' + _roles[0]

    @staticmethod
    def postprocess_wrong_relations(doc_node):
    
        duplicated_pairs = set()
        
        for relation_node in doc_node.findall('relation'):
            ne_type1, id1  = relation_node.find('infon[@key="role1"]').text.split('|', 1)
            ne_type2, id2  = relation_node.find('infon[@key="role2"]').text.split('|', 1)
            rel_type = relation_node.find('infon[@key="type"]').text
            pair = (ne_type1, id1, ne_type2, id2)
            
            if (ne_type1, ne_type2) in PubTator3Postprocessor._legal_pair_2_relation_types:
                if rel_type not in PubTator3Postprocessor._legal_pair_2_relation_types[(ne_type1, ne_type2)]:
                    relation_node.find('infon[@key="type"]').text = 'Association'
                elif (id1 in PubTator3Postprocessor._mutation_type_set) and (id2 not in PubTator3Postprocessor._mutation_type_set):
                    if rel_type not in PubTator3Postprocessor._legal_variant_relation_types:
                        relation_node.find('infon[@key="type"]').text = 'Association'
                elif (id2 in PubTator3Postprocessor._mutation_type_set) and (id1 not in PubTator3Postprocessor._mutation_type_set):
                    if rel_type not in PubTator3Postprocessor._legal_variant_relation_types:
                        relation_node.find('infon[@key="type"]').text = 'Association'
                elif (id2 in PubTator3Postprocessor._mutation_type_set) and (id1 in PubTator3Postprocessor._mutation_type_set):
                    relation_node.find('infon[@key="type"]').text = 'Association'
                
            if pair not in duplicated_pairs:
                duplicated_pairs.add(pair)
            else:
                doc_node.remove(relation_node)

    @staticmethod
    def remove_annotator_from_relation(doc_node):

        for relation_node in doc_node.findall('relation'):
            
            annotator = relation_node.find('infon[@key="annotator"]')
            
            if annotator != None:
                relation_node.remove(annotator)

    @staticmethod
    def update_relation_id(doc_node):
    
        bioc_id = 1
        relation_nodes = doc_node.findall('relation')
        for relation_node in relation_nodes:
            relation_node.attrib["id"] = "R" + str(bioc_id)
            bioc_id += 1

class Annotation:

    def __init__(self, 
                 ne_type,
                 identifier,
                 annotator,
                 updated_at,
                 text,
                 offset,
                 length):
        self.ne_type = ne_type
        self.identifier = identifier
        self.annotator = annotator
        self.updated_at = updated_at
        self.text = text
        self.offset = offset
        self.length = length
        
def update_bioc(in_bioc_xml_file, 
               chebi_id_mapping_dict,
               out_bioc_xml_file):
    print(in_bioc_xml_file)
    #print(protein_node_list)
    tree = ET.parse(in_bioc_xml_file)
    tree = tree.getroot()
    for passage_node in tree.findall('document/passage'):
        passage_offset = int(passage_node.find('offset').text)
        for infon_node in passage_node.findall('annotation/infon[@key="identifier"]'):
            #print(infon_node.text)
            if infon_node.text:
                has_star = True if str(infon_node.text[0]) == '*' else False
                id = infon_node.text.strip('*')
                if id in chebi_id_mapping_dict:
                    if id != chebi_id_mapping_dict[id]:
                        new_id = '*' + chebi_id_mapping_dict[id] if has_star else chebi_id_mapping_dict[id]
                        #print(in_bioc_xml_file, infon_node.text, new_id)
                        infon_node.text = new_id
    xml_writer = open(out_bioc_xml_file, 'w', encoding='utf8')
    xml_writer.write(ET.tostring(tree, encoding="unicode"))
    xml_writer.close()

def dump_text_instance_2_xml(
            text_instances, 
            annotation_id_2_ne_dict={},
            section='',
            ann_start_index=0,
            rel_start_index=0,
            offset_with_special_symbol_mapping=0,
            offset=0,
            ne_id_2_ne_type_dict={},
            pair_all_min_distance = {},
            pair_all_min_distance_num = {}):
            
    text = ''
    for text_ins in text_instances:
        text += ' ' + text_ins.text
    text = text.strip()
    
    out_str = ''
    
    out_str += '<passage>'
    out_str += '<infon key="type">' + section.replace('&', '&#38;').replace('<', '&lt;').replace('>', '&gt;').replace('\'', '&#39;').replace('\"', '&#34;') + '</infon>'
    out_str += '<offset>' + str(offset) + '</offset>'
    out_str += '<text>' + text.replace('&', '&#38;').replace('<', '&lt;').replace('>', '&gt;').replace('\'', '&#39;').replace('\"', '&#34;') + '</text>'
    
    annotation_2_ann_id = {}
    
    for text_instance in text_instances:
        for ann in text_instance.annotations:
            ann_id = str(ann_start_index)
            ann.ids = list(ann.ids)
            ann.ids.sort()
            #position_with_speical_symbol_mapping = len(text_instance.text[:ann.position].replace('<', '&lt;').replace('>', '&gt;').replace('&', '&#38;').replace('\'', '&#39;').replace('\"', '&#34;'))
            #ann_length_with_speical_symbol_mapping = len(ann.text.replace('<', '&lt;').replace('>', '&gt;').replace('&', '&#38;').replace('\'', '&#39;').replace('\"', '&#34;'))
            out_str += '<annotation id="' + ann_id + '">'
            out_str += '<infon key="type">' + ann.ne_type + '</infon>'
            
            _ids = []
            # in case "tmVar:<150fs;VariantGroup:0;RS#:150f"
            for _id in ann.ids:
                _ids.append(_id.replace('&', '&#38;').replace('<', '&lt;').replace('>', '&gt;').replace('\'', '&#39;').replace('\"', '&#34;'))
            out_str += '<infon key="identifier">' + ';'.join(_ids) + '</infon>'
            #xml_writer.write('<location offset="' + str(offset_with_special_symbol_mapping + position_with_speical_symbol_mapping) + '" length="' + str(ann_length_with_speical_symbol_mapping) + '"/>')
            out_str += '<location offset="' + str(offset + ann.position) + '" length="' + str(ann.length) + '"/>'
            out_str += '<text>' + ann.text.replace('&', '&#38;').replace('<', '&lt;').replace('>', '&gt;').replace('\'', '&#39;').replace('\"', '&#34;') + '</text>'
            out_str += '</annotation>'
            annotation_2_ann_id[ann] = ann_id
            
            annotation_id_2_ne_dict[ann_id] = ann
            ann_start_index += 1
            for id in ann.ids:
                ne_id_2_ne_type_dict[id] = ann.ne_type
        offset += len(text_instance.text) + 1
        offset_with_special_symbol_mapping += len(text_instance.text.replace('&', '&#38;').replace('<', '&lt;').replace('>', '&gt;').replace('\'', '&#39;').replace('\"', '&#34;')) + 1
    #raise('GG')
    ann_id_2_sent_index = defaultdict(int)
    ann_id_2_ne_id = {}
    for i, text_instance in enumerate(text_instances):
        for ann in text_instance.annotations:
            ann_id = annotation_2_ann_id[ann]
            ann_id_2_sent_index[ann_id] = i
            ann_id_2_ne_id[ann_id] = ann.ids
    #for ann_id in ann_id_2_sent_index.keys():
    #    ann_id_2_sent_index[ann_id].sort()
    for ann_id1, index1 in ann_id_2_sent_index.items():
        for ne_id1 in ann_id_2_ne_id[ann_id1]:
            for ann_id2, index2 in ann_id_2_sent_index.items():
                for ne_id2 in ann_id_2_ne_id[ann_id2]:
                    if ann_id1 != ann_id2:
                        pair = (ne_id1, ne_id2)
                        min_distance = 1000
                        if pair not in pair_all_min_distance:
                            pair_all_min_distance[pair] = set()
                            pair_all_min_distance[pair].add((ann_id1, ann_id2))
                            pair_all_min_distance_num[pair] = abs(index2 - index1)
                        elif abs(index2 - index1) < pair_all_min_distance_num[pair]:
                            pair_all_min_distance[pair] = set()
                            pair_all_min_distance[pair].add((ann_id1, ann_id2))
                            pair_all_min_distance_num[pair] = abs(index2 - index1)
                        elif (abs(index2 - index1) == min_distance):
                            pair_all_min_distance[pair].add((ann_id1, ann_id2))
           
    out_str += '</passage>'
    
    return ann_start_index, rel_start_index, offset_with_special_symbol_mapping, offset, out_str

def dump_document_2_xml_with_instance(
            document,
            rel_pair_list):

    annotation_id_2_ne_dict={}
    ne_id_2_ne_type_dict={}
    
    out_str = ''
    
    out_str += '<document>'
    out_str += '<id>' + document.id + '</id>'
    title_list = []
    abstract_list = []
    for text_ins in document.text_instances:
        # 'TITLE' for bioc full text
        if text_ins.section == 'title' or text_ins.section == 'TITLE':
            title_list.append(text_ins)
        else:
            abstract_list.append(text_ins)
    pair_all_min_distance = {}
    pair_all_min_distance_num = {}
    ann_start_index, rel_start_index, offset_with_special_symbol_mapping, offset = 0, 0, 0, 0
    if len(title_list) > 0:
        ann_start_index, rel_start_index, offset_with_special_symbol_mapping, offset, _out_str = dump_text_instance_2_xml(
            title_list, 
            annotation_id_2_ne_dict=annotation_id_2_ne_dict,
            section='title',
            ann_start_index=0,
            rel_start_index=0,
            offset_with_special_symbol_mapping=0,
            offset=0,
            ne_id_2_ne_type_dict=ne_id_2_ne_type_dict,
            pair_all_min_distance=pair_all_min_distance,
            pair_all_min_distance_num=pair_all_min_distance_num)
        out_str += _out_str
    else:
        print(document.id + ' len(title_list) == 0')
    if len(abstract_list) > 0:
        ann_start_index, rel_start_index, offset_with_special_symbol_mapping, offset, _out_str = dump_text_instance_2_xml(
            abstract_list, 
            annotation_id_2_ne_dict=annotation_id_2_ne_dict,
            section='abstract',
            ann_start_index=ann_start_index,
            rel_start_index=rel_start_index,
            offset_with_special_symbol_mapping=offset_with_special_symbol_mapping,
            offset=offset,
            ne_id_2_ne_type_dict=ne_id_2_ne_type_dict,
            pair_all_min_distance=pair_all_min_distance,
            pair_all_min_distance_num=pair_all_min_distance_num)
        out_str += _out_str
    else:
        print(document.id + ' len(abstract_list) == 0')
    
    counter = 0
    for i, rel_pair in enumerate(rel_pair_list):
        id1 = rel_pair.id1
        id2 = rel_pair.id2
        print(f'Processing relation {i}: {id1} - {id2} {rel_pair.rel_type} {rel_pair.direction_type}')
        if (id1 not in ne_id_2_ne_type_dict) or (id2 not in ne_id_2_ne_type_dict):
            continue
        ne_type1 = ne_id_2_ne_type_dict[id1]   
        ne_type2 = ne_id_2_ne_type_dict[id2]

        if rel_pair.direction_type == 'Left_to_Right':
            role1 = 'Subject'
            role2 = 'Object'
        elif rel_pair.direction_type == 'Right_to_Left':
            role1 = 'Object'
            role2 = 'Subject'
        else:
            role1 = ''
            role2 = ''
        
        if ne_type1 in PubTator3Postprocessor._mutation_type_set:
            _ne_type1 = 'Variant'
        else:
            _ne_type1 = ne_type1
        if ne_type2 in PubTator3Postprocessor._mutation_type_set:
            _ne_type2 = 'Variant'
        else:
            _ne_type2 = ne_type2
        
        if _ne_type1 > _ne_type2:
            ne_type1, ne_type2, id1, id2, role1, role2 = ne_type2, ne_type1, id2, id1, role2, role1
            
        if _ne_type1 == _ne_type2 and id1 > id2:
            ne_type1, ne_type2, id1, id2, role1, role2 = ne_type2, ne_type1, id2, id1, role2, role1
            
        
        out_str += '<relation id="R' + str(i) + '">'
        
        out_str += '<infon key="type">' + rel_pair.rel_type + '</infon>'
        out_str += '<infon key="novelty">' + rel_pair.novelty_type + '</infon>'
        #if (note_id1 != chebi_id1) or (note_id2 != chebi_id2):
        #    xml_writer.write('<infon key="note">' + note_id1 + '|' + note_id2 + '</infon>')
        out_str += '<infon key="score">' + str(round(float(rel_pair.rel_score), 4)) + '</infon>'
        out_str += '<infon key="novelty_score">' + str(round(float(rel_pair.novelty_score), 4)) + '</infon>'
        out_str += '<infon key="direction_score">' + str(round(float(rel_pair.direction_score), 4)) + '</infon>'
        
        # tmVar:<150fs;VariantGroup:0;RS#:150f
        _id1 = id1.replace('&', '&#38;').replace('<', '&lt;').replace('>', '&gt;').replace('\'', '&#39;').replace('\"', '&#34;')
        _id2 = id2.replace('&', '&#38;').replace('<', '&lt;').replace('>', '&gt;').replace('\'', '&#39;').replace('\"', '&#34;')
        
        out_str += f'<infon key="role1" role="{role1}">{ne_type1}|{_id1}</infon>'
        out_str += f'<infon key="role2" role="{role2}">{ne_type2}|{_id2}</infon>'
        
        if (id1, id2) in pair_all_min_distance.keys():
            _pairs = list(pair_all_min_distance[(id1, id2)])
            _pairs.sort()
            for j, (index1, index2) in enumerate(_pairs):
                out_str += '<node refid="' + str(counter) + '" role="' + str(index1) + ',' + str(index2) + '"/>'
                counter += 1
        elif (id2, id1) in pair_all_min_distance.keys():
            _pairs = list(pair_all_min_distance[(id2, id1)])
            _pairs.sort()
            for j, (index1, index2) in enumerate(_pairs):
                out_str += '<node refid="' + str(counter) + '" role="' + str(index2) + ',' + str(index1) + '"/>'
                counter += 1
        out_str += '</relation>'
        rel_start_index += 1
    
    out_str += '</document>'
    
    return out_str

def dump_documents_2_xml_with_instance(
    in_xml_file,
    documents,
    out_xml_file,
    pmid_2_rel_pair_dict):    
    
    out_str = ''
    out_str += '<?xml version="1.0" encoding="UTF-8"?>'
    out_str += '<!DOCTYPE collection SYSTEM "BioC.dtd">'
    out_str += '<collection>'
    out_str += '<source>PubMed</source>'
    out_str += '<date>' + str(datetime.date.today()) + '</date>'
    out_str += '<key>collection.key</key>'
        
    for document in documents:
        #print(document.id, len(pmid_2_rel_pair_dict), len(pmid_2_rel_pair_dict[document.id]))
        #print(pmid_2_rel_pair_dict.keys())
        out_str += dump_document_2_xml_with_instance(
            document,
            pmid_2_rel_pair_dict[document.id])
        
    out_str += '</collection>'
    
    if os.path.exists(out_xml_file):
        return
    
    convert_bioc_to_pubtator3_xml(
        in_orig_xml_file = in_xml_file,
        rel_xml_str      = out_str,
        out_xml_file     = out_xml_file)
        
def convert_bioc_to_pubtator3_xml(in_orig_xml_file,
                                  rel_xml_str,
                                  out_xml_file):
    
    orig_tree = None
    rel_tree = None
    try:
        orig_tree = ET.parse(in_orig_xml_file)
    except:
        print('GG', in_orig_xml_file)
        return      
    
    if os.path.exists(out_xml_file):
        return
    try:
        rel_tree = ET.ElementTree(ET.fromstring(rel_xml_str))
    except:
        return
    
    _orig_tree = orig_tree.getroot()
    _rel_tree = rel_tree.getroot()
    
    if os.path.exists(out_xml_file):
        return
    
    orig_doc_dict = {}
    for orig_doc_node in _orig_tree.findall('document'):
        pmid = orig_doc_node.find('id').text
        orig_doc_dict[pmid] = orig_doc_node
    
    for rel_doc_node in _rel_tree.findall('document'):
        
        pmid = rel_doc_node.find('id').text
        
        if pmid not in orig_doc_dict:
            continue
        
        orig_doc_node = orig_doc_dict[pmid]
        
        mutation_2_normalized_id_dict, mutation_2_gene_id_dict = PubTator3Postprocessor.get_mutation_2_normalized_id_dict(rel_doc_node)
        
        PubTator3Postprocessor.add_gene_disease_via_mutation(rel_doc_node, mutation_2_gene_id_dict)
        PubTator3Postprocessor.remove_redundant_mutation_relation(rel_doc_node, mutation_2_normalized_id_dict)
        PubTator3Postprocessor.normalize_mutation_id(rel_doc_node, mutation_2_normalized_id_dict)
        PubTator3Postprocessor.update_relation_order(rel_doc_node)
        PubTator3Postprocessor.postprocess_wrong_relations(rel_doc_node)
        PubTator3Postprocessor.remove_annotator_from_relation(rel_doc_node)
        PubTator3Postprocessor.update_relation_id(rel_doc_node)
        
        ann_id_to_index_dict = {}
        for ann_node in rel_doc_node.findall('passage/annotation'):
            ann_id = ann_node.attrib["id"]
            loc_node = ann_node.find('location')
            if ann_node.find('infon[@key="identifier"]'):
                ne_ids = list(ann_node.find('infon[@key="identifier"]').text.split(';'))
                ne_ids.sort()
                ne_ids = ';'.join(ne_ids)
            else:
                ne_ids = ''
            ann_id_to_index_dict[ann_id] = loc_node.attrib["offset"] + '|' + loc_node.attrib["length"] + '|' + ne_ids
            
        ann_index_2_orig_ann_id_dict = {}
        extended_ann_index_2_orig_ann_id_dict = {}
        ann_index_2_orig_ne_type_dict = {}
        extended_ann_index_2_orig_ne_type_dict = {}
        for ann_node in orig_doc_node.findall('passage/annotation'):
            ann_id = ann_node.attrib["id"]
            loc_node = ann_node.find('location')                
            if ann_node.find('infon[@key="identifier"]'):
                ne_ids = list(ann_node.find('infon[@key="identifier"]').text.split(';'))
                ne_ids.sort()
                ne_ids = ';'.join(ne_ids)
            else:
                ne_ids = ''
            ne_type = ann_node.find('infon[@key="type"]').text
            #ann_id_to_index_dict[ann_id] = loc_node.attrib["offset"] + '|' + loc_node.attrib["length"] + '|' + ne_ids
            ann_index_2_orig_ann_id_dict[loc_node.attrib["offset"] + '|' + loc_node.attrib["length"] + '|' + ne_ids] = ann_id
            extended_ann_index_2_orig_ann_id_dict[str(int(loc_node.attrib["offset"]) - 1) + '|' + loc_node.attrib["length"] + '|' + ne_ids] = ann_id
            ann_index_2_orig_ne_type_dict[loc_node.attrib["offset"] + '|' + loc_node.attrib["length"] + '|' + ne_ids] = ne_type
            extended_ann_index_2_orig_ne_type_dict[str(int(loc_node.attrib["offset"]) - 1) + '|' + loc_node.attrib["length"] + '|' + ne_ids] = ne_type
            
        rel_ann_id_2_orig_ann_id = {}
        rel_ann_id_2_orig_ne_type = {}
        for rel_ann_id, ann_index in ann_id_to_index_dict.items():
            if ann_index in ann_index_2_orig_ann_id_dict:
                rel_ann_id_2_orig_ann_id[rel_ann_id] = ann_index_2_orig_ann_id_dict[ann_index]
            elif ann_index in extended_ann_index_2_orig_ann_id_dict:
                rel_ann_id_2_orig_ann_id[rel_ann_id] = extended_ann_index_2_orig_ann_id_dict[ann_index]
            
            if ann_index in ann_index_2_orig_ne_type_dict:
                rel_ann_id_2_orig_ne_type[rel_ann_id] = ann_index_2_orig_ne_type_dict[ann_index]
            elif ann_index in extended_ann_index_2_orig_ne_type_dict:
                rel_ann_id_2_orig_ne_type[rel_ann_id] = extended_ann_index_2_orig_ne_type_dict[ann_index]
        refid = 0
        
        for rel_node in rel_doc_node.findall('relation'):
            new_rel_node = copy.deepcopy(rel_node)
            
            
        for rel_node in rel_doc_node.findall('relation'):
            new_rel_node = copy.deepcopy(rel_node)
            for node in new_rel_node.findall('node'):
                new_rel_node.remove(node)

            orig_ne_type1, orig_ne_type2 = '', ''
            for node in rel_node.findall('node'):
                role1, role2 = node.attrib["role"].split(',')
                try:
                    orig_role1, orig_role2 = rel_ann_id_2_orig_ann_id[role1], rel_ann_id_2_orig_ann_id[role2]
                    orig_ne_type1, orig_ne_type2 = rel_ann_id_2_orig_ne_type[role1], rel_ann_id_2_orig_ne_type[role2]
                    #print(pmid, role1, role2, orig_role1, orig_role2)
                    #attrib = {'refid': str(refid), "role":"\"" + orig_role1 + ',' + orig_role2 + "\""}
                    #new_rel_node.makeelement('node', attrib)
                    new_node = ET.Element('node')
                    new_node.set('refid', str(refid))
                    new_node.set('role', orig_role1 + ',' + orig_role2)
                    new_rel_node.append(new_node)
                    refid += 1
                except:
                    print(out_xml_file)
                    print(pmid)
                    print(role1 in rel_ann_id_2_orig_ann_id, role1)
                    print(role2 in rel_ann_id_2_orig_ann_id, role2)
            if orig_ne_type1 == '' or orig_ne_type2 == '':
                continue
            new_rel_node.find('infon[@key="role1"]').text = orig_ne_type1 + '|' + new_rel_node.find('infon[@key="role1"]').text.split('|', 1)[1]
            new_rel_node.find('infon[@key="role2"]').text = orig_ne_type2 + '|' + new_rel_node.find('infon[@key="role2"]').text.split('|', 1)[1]
            orig_doc_node.append(new_rel_node)
    
    out_str = "<?xml version='1.0' encoding='UTF-8'?><!DOCTYPE collection SYSTEM \'BioC.dtd\'>"+ ET.tostring(orig_tree.getroot(), encoding="unicode")
    
    if not os.path.exists(out_xml_file):
        with open(out_xml_file, 'w', encoding='utf8') as xml_writer:
            xml_writer.write(out_str)