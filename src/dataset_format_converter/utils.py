# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 14:09:51 2021

@author: laip2
"""
import re
from document import TextInstance
import json
import logging
from sentence_spliter import regex_sentence_boundary_gen
        
logger = logging.getLogger(__name__)

def _spacy_split_sentence(text, nlp):
    offset = 0
    offsets = []
    doc = nlp(text)
    
    do_not_split = False
    start = 0
    end = 0
    for sent in doc.sents:
        if re.search(r'\b[a-z]\.$|[A-Z] ?\>$|[^a-z]del\.$| viz\.$', sent.text):
            if not do_not_split:
                start = offset
            end = offset + len(sent.text)
            offset = end
            for c in text[end:]:
                if c == ' ':
                    offset += 1
                else:
                    break
            do_not_split = True
        else:
            if do_not_split:                
                do_not_split = False
                end = offset + len(sent.text)
                offset = end
                for c in text[end:]:
                    if c == ' ':
                        offset += 1
                    else:
                        break
                offsets.append((start, end))
            else:
                start = offset
                end = offset + len(sent.text)
                offsets.append((start, end))
                
                offset = end
                for c in text[end:]:
                    if c == ' ':
                        offset += 1
                    else:
                        break
        
    if do_not_split:
        offsets.append((start, end))
    return offsets

def split_sentence(document, nlp = None):
    new_text_instances = []
    for text_instance in document.text_instances:
        
        #offsets = [o for o in _nltk_split_sentence(text_instance.text)]
        if nlp == None:
            offsets = [o for o in regex_sentence_boundary_gen(text_instance.text)]
        else:
            offsets = [o for o in _spacy_split_sentence(text_instance.text, nlp)]
            
        _tmp_text_instances = []
        for start, end in offsets:
            new_text_instance = TextInstance(text_instance.text[start:end])
            new_text_instance.offset = start
            new_text_instance.section = text_instance.section
            _tmp_text_instances.append(new_text_instance)
        for annotation in text_instance.annotations:
            is_entity_splited = True
            for _tmp_text_instance in _tmp_text_instances:
                if _tmp_text_instance.offset <= annotation.position and \
                    (annotation.position + annotation.length) - _tmp_text_instance.offset <= len(_tmp_text_instance.text):
                    annotation.position = annotation.position - _tmp_text_instance.offset
                    _tmp_text_instance.annotations.append(annotation)
                    is_entity_splited = False
                    break
            if is_entity_splited:
                print(annotation.position, annotation.length, annotation.text)
                print (' splited by Spacy\' sentence spliter is failed to be loaded into TextInstance\n')
                for _tmp_text_instance in _tmp_text_instances:
                    print (_tmp_text_instance.offset, len(_tmp_text_instance.text), _tmp_text_instance.text)
        new_text_instances.extend(_tmp_text_instances)
    
    document.text_instances = new_text_instances

def tokenize_document_by_bert(document, tokenizer):
    for text_instance in document.text_instances:
         
        text_instance.tokens = tokenizer.tokenize(text_instance.text)
        for entity in text_instance.annotations:
            start_wo_space = len(text_instance.text[:entity.position].replace(' ', ''))
            end_wo_space = start_wo_space + len(entity.text.replace(' ', ''))
            entity.start_token = -1
            entity.end_token = -1
            previous_wo_space = 0
            for i, token in enumerate(text_instance.tokens):
                #print(i, token, entity.start_token, entity.end_token, previous_wo_space)
                clean_token = re.sub(r'^##', '', token)
                
                if entity.start_token == -1 and previous_wo_space <= start_wo_space and start_wo_space < previous_wo_space + len(clean_token):
                    entity.start_token = i
                if previous_wo_space <= end_wo_space and end_wo_space <= previous_wo_space + len(clean_token):
                    entity.end_token = i
                    break
                previous_wo_space += len(clean_token)

            if entity.start_token == -1 or entity.end_token == -1:
                print('entity.start_token', entity.start_token)
                print('entity.end_token', entity.end_token)
                print('start_wo_space', start_wo_space)
                print('end_wo_space', end_wo_space)
                print('text_instance.text', text_instance.text)
                print('text_instance.tokens', text_instance.tokens)
                print('entity.text', entity.text)
                raise('Error: entity.start_token or entity.end_token is not found')
            
def split_documents(documents, tokenizer):    

    for document in documents:
        split_sentence(document)
        tokenize_document_by_bert(document, tokenizer)

def get_ne_id_2_ne_text_dict(document):
    ne_id_2_ne_text_dict = {}
    for text_instance in document.text_instances:
        for ann in text_instance.annotations:
            for id in ann.ids:
                ne_id_2_ne_text_dict[id] = ann.text
    return ne_id_2_ne_text_dict

def dump_documents_2_zs_instruction_format(
    all_documents,
    out_bert_file,
    considered_ne_pairs):
    
    out_writer = open(out_bert_file, 'w', encoding='utf8')
    first_record = True
    for document in all_documents:
        pmid = document.id
        
        id_and_ne_types = set()
        for i, text_instance in enumerate(document.text_instances):
            for ann in text_instance.annotations:
                for id in ann.ids:
                    id_and_ne_types.add((id, ann.ne_type))

        for id1, ne_type1 in id_and_ne_types:
            for id2, ne_type2 in id_and_ne_types:
                if id1 >= id2:
                    continue
                
                #for (id1, id2) in document.relation_pairs:
                #    print(pmid, id1, id2, document.relation_pairs[(id1, id2)])

                re_label, nov_label, subj_id = 'None', 'None', 'None'
                if (id1, id2) in document.relation_pairs:
                    _tks = document.relation_pairs[(id1, id2)].split('\t')
                    if len(_tks) == 1:
                        re_label = _tks[0]
                    elif len(_tks) == 2:
                        re_label, nov_label = _tks
                    elif len(_tks) == 3:
                        re_label, nov_label, subj_id = _tks
                elif (id2, id1) in document.relation_pairs:
                    _tks = document.relation_pairs[(id2, id1)].split('\t')
                    if len(_tks) == 1:
                        re_label = _tks[0]
                    elif len(_tks) == 2:
                        re_label, nov_label = _tks
                    elif len(_tks) == 3:
                        re_label, nov_label, subj_id = _tks
                if ((ne_type1, ne_type2) not in considered_ne_pairs) and ((ne_type2, ne_type1) not in considered_ne_pairs):
                    continue

                #print(pmid, ne_type1, ne_type2, id1, id2, re_label, nov_label, subj_id)


                ''' PMID "\t" + 
                ne_type1 "\t" +
                ne_type2 "\t" +
                ID1 "\t" +
                ID2 "\t" +
                instruction + "\t"
                context + "\t"
                response + "\t"
                '''
                
                if ne_type1 < ne_type2:
                    ne_type1_suffix = 'Src'
                    ne_type2_suffix = 'Tgt'
                elif ne_type1 > ne_type2:
                    ne_type1_suffix = 'Tgt'
                    ne_type2_suffix = 'Src'
                else:
                    ne_type1_suffix = ''
                    for i, text_instance in enumerate(document.text_instances):
                        for ann in text_instance.annotations:
                            ne_type_suffix = ''
                            if id1 in ann.ids:
                                ne_type1_suffix = 'Src'
                                ne_type2_suffix = 'Tgt'
                                break
                            elif id2 in ann.ids:
                                ne_type1_suffix = 'Tgt'
                                ne_type2_suffix = 'Src'
                                break
                        if ne_type1_suffix != '':
                            break
                if len(considered_ne_pairs) == 1:
                    ne_type1_suffix = ''
                    ne_type2_suffix = ''
                sents = []
                ne1_text = ''
                ne2_text = ''
                ne1_tag = ''
                ne2_tag = ''
                for i, text_instance in enumerate(document.text_instances):
                    tagged_sent = [str(c) for c in text_instance.text]

                    for ann in text_instance.annotations:
                        ne_type_suffix = ''
                        if id1 in ann.ids:
                            ne_type_suffix = ne_type1_suffix
                            if ne1_text == '':
                                ne1_text = ann.text
                        elif id2 in ann.ids:
                            ne_type_suffix = ne_type2_suffix
                            if ne2_text == '':
                                ne2_text = ann.text
                        else:
                            continue

                        _ne_type = ann.ne_type
                        if _ne_type == 'SequenceVariant':
                            _ne_type = 'GeneOrGeneProduct'

                        tagged_sent[ann.position] = '<' + _ne_type + ne_type_suffix + '> ' + tagged_sent[ann.position]
                        tagged_sent[ann.position + ann.length - 1] = tagged_sent[ann.position + ann.length - 1] + ' </' + _ne_type + ne_type_suffix + '>'
                    
                    sents.append(''.join(tagged_sent))
                
                if len(considered_ne_pairs) == 1:

                    if ne_type1 == 'Chemical':
                        ne1_text, ne2_text = ne2_text, ne1_text

                    instruction = '## Instructions\n\nPlease read the below passages, and then the question regarding to the highlighted pair, "' + ne1_text + '" and "' + ne2_text + '", and respond in JSON format.'
                    content  = '### Passage\n\n' + '\n\n'.join(sents)
                    #response = 'For the highlighted pair ' + ne1_text + ' and ' + ne2_text + ':\n' + 'BioRED relation label: ' + re_label + '\n' + 'Direction label: ' + dir_label + '\n' + 'Novelty label: ' + nov_label
       
                    q_prompt_text = '### Yes/No Question\n\n'
                    q_prompt_text += 'According the above passage, is the disease **'
                    q_prompt_text += ne1_text + '** induced by the chemical **'
                    q_prompt_text += ne2_text + '**?"\n\n'
                    q_prompt_text += '### Response Format (JSON Example)\n\n'
                    q_prompt_text += 'After carefully reading the descriptions of the two entities and considering their biological relationship, use a chain-of-thought approach to reason through the relationship step by step. Reflect on the relationship, directionality, and novelty, explaining your reasoning process clearly. Then, respond to all questions in the following simplified JSON format:\n\n'
                    q_prompt_text += '''```json
{
  "Yes/No Question": "Yes",
  "Explanation 1": "Based on the passage, ....",  
}
```'''
                    response = json.dumps({
                        "relation_label": re_label
                    })
                else:

                    instruction = '## Instructions\n\nPlease read the below passages, and then identify the BioRED relation, direction, and novelty labels for the highlighted pair, "' + ne1_text + '" and "' + ne2_text + '", and respond in JSON format.'
                    content  = '### Passage\n\n' + '\n\n'.join(sents)
                    #response = 'For the highlighted pair ' + ne1_text + ' and ' + ne2_text + ':\n' + 'BioRED relation label: ' + re_label + '\n' + 'Direction label: ' + dir_label + '\n' + 'Novelty label: ' + nov_label

                    q_prompt_text = '### Questions\n\n'
                    q_prompt_text += '#### Question 1 (BioRED relation label):\n'
                    q_prompt_text += 'Please select an option number (1~14) below that best describes the biological relationship between **'
                    q_prompt_text += ne1_text + '** and **'
                    q_prompt_text += ne2_text + '**."\n\n'
                    q_prompt_text += '1. ASSOCIATE: The relation is closedly associated with a biologival event, indicating its potential impact on biological processes. Note: There must be a clear and explicit biologival event.\n'
                    q_prompt_text += '2. CAUSE: A positive correlation exists when the status of one entity tends to increase (or decrease) as the other increase (or decreases). This type includes chemical-induced diseases and genetic diseases caused by variants.\n'
                    q_prompt_text += '3. COMPARE: The effect comparison of two chemicals/drugs.\n'
                    q_prompt_text += '4. CONVERT: A chemical is converted into another chemical by a chemical reaction.\n'
                    q_prompt_text += '5. COTREAT: It is defined as the use of 2 or more drugs administered separately or in a fixed-dose combination.\n'
                    q_prompt_text += '6. DRUG_INTERACT: A pharmacodynamic interaction between two chemicals that results in an array of side effects.\n'
                    q_prompt_text += '7. INHIBIT: A negative correlation exists when the status of the two entities tends to be opposite. This type includes disease-gene and chemical-variant.\n'
                    q_prompt_text += '8. INTERACT: Physical interaction, like protein-binding.\n'
                    q_prompt_text += '9. NEGATIVE_CORRELATE: A negative correlation exists when the status of the two entities tends to be opposite. This type includes chemical-gene, chemical co-expression, and gene co-expression.\n'
                    q_prompt_text += '10. POSITIVE_CORRELATE: A positive correlation exists when the status of one entity tends to increase (or decrease) as the other increase (or decreases). This type includes chemical-gene, chemical co-expression, and gene co-expression.\n'
                    q_prompt_text += '11. PREVENT: A negative correlation exists when the status of the two entities tends to be opposite. This type includes variant-disease.\n'
                    q_prompt_text += '12. STIMULATE: A positive correlation exists when the status of one entity tends to increase (or decrease) as the other increase (or decreases). This type includes disease-gene and disease-variant.\n'
                    q_prompt_text += '13. TREAT: A chemical/drug treats a disease.\n'
                    q_prompt_text += '14. None: None of the above. Note: This has also included some general types of relationships such as is-a, part-of, and has-a etc.\n\n'
                    q_prompt_text += '#### Question 2 (BioRED direction label):\n'
                    q_prompt_text += 'Which of these options (1-3) best describes the novelty between **'
                    q_prompt_text += ne1_text + '** and **'
                    q_prompt_text += ne2_text + '**?"\n\n'
                    q_prompt_text += '1. NOVEL: It is used for relations that are related to the main point or novelty of the abstract. Any information that would be part of the results or conclusions of the paper is considered novel.\n'
                    q_prompt_text += '2. NO: It is for relations that are background information, typically providing context for the abstract, such as results of previous studies or relevant details that are needed to understand why the paper is important.\n'
                    q_prompt_text += '3. None: None of the above.\n\n'
                    q_prompt_text += '#### Question 3 (BioRED novelty label):\n'
                    q_prompt_text += 'Following the Question 1, which of these options (1-3) is the subject of the pair , **'
                    q_prompt_text += ne1_text + '** and **'
                    q_prompt_text += ne2_text + '**?"\n'
                    q_prompt_text += '1. First: **' + ne1_text + '** is the subject of the biological relationship, and **' + ne2_text + '** is the object.\n'
                    q_prompt_text += '2. Second: **' + ne2_text + '** is the subject of the biological relationship, and **' + ne1_text + '** is the object.\n'
                    q_prompt_text += '3. No_direction: A biological relationship exists, but the directionality of this relationship is not clearly defined in the passage."\n'
                    q_prompt_text += '4. None: None of the above or not clear.\n'
                    q_prompt_text += '### Response Format (JSON Example)\n\n'
                    q_prompt_text += 'After carefully reading the descriptions of the two entities and considering their biological relationship, use a chain-of-thought approach to reason through the relationship step by step. Reflect on the relationship, directionality, and novelty, explaining your reasoning process clearly. Then, respond to all questions in the following simplified JSON format:\n\n'
                    q_prompt_text += '''```json
{
  "Question 1": "10. POSITIVE_CORRELATE",
  "Explanation 1": "Based on the passage, 'cmvIL-10' and 'SOCS3' show a positive correlation, as the activation of STAT3 by cmvIL-10 leads to the expression of SOCS3, indicating that as one increases, so does the other.",
  
  "Question 2": "1. NOVEL",
  "Explanation 2": "This relationship appears to be novel because the passage highlights new findings regarding the effect of IL-10R1 variants on cmvIL-10 and SOCS3 expression, which are part of the study’s conclusions.",
  
  "Question 3": "1. First",
  "Explanation 3": "'cmvIL-10' is the subject as it induces the expression of SOCS3, making SOCS3 the object in this biological relationship."
}
```'''

                    direction_label = 'Rightward' if id1 == subj_id else 'Leftward'
                    if subj_id == 'None' or subj_id == '':
                        if re_label != 'None':
                            direction_label = 'No_Direct'
                        else:
                            direction_label = 'None'
                    
                    #response = 'For the highlighted pair ' + ne1_text + ' and ' + ne2_text + ':\n' + 'BioRED relation label: ' + re_label + '\n' + 'Direction label: ' + dir_label + '\n' + 'Novelty label: ' + nov_label
                    response = json.dumps({
                        "BioRED_relation_label": re_label,
                        "direction_label": direction_label,
                        "novelty_label": nov_label
                    })


                messages = []
                messages.append({"role": "system", "content": "You are a bioinformatics expert."})
                messages.append({"role": "user", "content": instruction + "\n" + content + "\n" + q_prompt_text})
                messages.append({"role": "assistant", "content": response})
                #messages.append({"prompt": instruction + "\n" + content})
                #messages.append({"completion": response})

                
                json.dump({"messages": messages}, out_writer, ensure_ascii=False)
                out_writer.write('\n')
                out_writer.flush()
    out_writer.close()

def dump_documents_2_instruction_format(
    all_documents,
    out_bert_file,
    considered_ne_pairs):
    
    out_writer = open(out_bert_file, 'w', encoding='utf8')
    first_record = True
    for document in all_documents:
        pmid = document.id
        
        id_and_ne_types = set()
        for i, text_instance in enumerate(document.text_instances):
            for ann in text_instance.annotations:
                for id in ann.ids:
                    id_and_ne_types.add((id, ann.ne_type))

        for id1, ne_type1 in id_and_ne_types:
            for id2, ne_type2 in id_and_ne_types:
                if id1 >= id2:
                    continue
                
                #for (id1, id2) in document.relation_pairs:
                #    print(pmid, id1, id2, document.relation_pairs[(id1, id2)])

                re_label, nov_label, subj_id = 'None', 'None', 'None'
                if (id1, id2) in document.relation_pairs:
                    _tks = document.relation_pairs[(id1, id2)].split('\t')
                    if len(_tks) == 1:
                        re_label = _tks[0]
                    elif len(_tks) == 2:
                        re_label, nov_label = _tks
                    elif len(_tks) == 3:
                        re_label, nov_label, subj_id = _tks
                elif (id2, id1) in document.relation_pairs:
                    _tks = document.relation_pairs[(id2, id1)].split('\t')
                    if len(_tks) == 1:
                        re_label = _tks[0]
                    elif len(_tks) == 2:
                        re_label, nov_label = _tks
                    elif len(_tks) == 3:
                        re_label, nov_label, subj_id = _tks
                if ((ne_type1, ne_type2) not in considered_ne_pairs) and ((ne_type2, ne_type1) not in considered_ne_pairs):
                    continue

                ''' PMID "\t" + 
                ne_type1 "\t" +
                ne_type2 "\t" +
                ID1 "\t" +
                ID2 "\t" +
                instruction + "\t"
                context + "\t"
                response + "\t"
                '''
                
                if ne_type1 < ne_type2:
                    ne_type1_suffix = 'Src'
                    ne_type2_suffix = 'Tgt'
                elif ne_type1 > ne_type2:
                    ne_type1_suffix = 'Tgt'
                    ne_type2_suffix = 'Src'
                else:
                    ne_type1_suffix = ''
                    for i, text_instance in enumerate(document.text_instances):
                        for ann in text_instance.annotations:
                            ne_type_suffix = ''
                            if id1 in ann.ids:
                                ne_type1_suffix = 'Src'
                                ne_type2_suffix = 'Tgt'
                                break
                            elif id2 in ann.ids:
                                ne_type1_suffix = 'Tgt'
                                ne_type2_suffix = 'Src'
                                break
                        if ne_type1_suffix != '':
                            break

                if len(considered_ne_pairs) == 1:
                    ne_type1_suffix = ''
                    ne_type2_suffix = ''
                    
                sents = []
                ne1_text = ''
                ne2_text = ''
                ne1_tag = ''
                ne2_tag = ''
                for i, text_instance in enumerate(document.text_instances):
                    tagged_sent = [str(c) for c in text_instance.text]

                    for ann in text_instance.annotations:
                        ne_type_suffix = ''
                        if id1 in ann.ids:
                            ne_type_suffix = ne_type1_suffix
                            if ne1_text == '':
                                ne1_text = ann.text
                        elif id2 in ann.ids:
                            ne_type_suffix = ne_type2_suffix
                            if ne2_text == '':
                                ne2_text = ann.text
                        else:
                            continue

                        _ne_type = ann.ne_type
                        if _ne_type == 'SequenceVariant':
                            _ne_type = 'GeneOrGeneProduct'

                        tagged_sent[ann.position] = '<' + _ne_type + ne_type_suffix + '> ' + tagged_sent[ann.position]
                        tagged_sent[ann.position + ann.length - 1] = tagged_sent[ann.position + ann.length - 1] + ' </' + _ne_type + ne_type_suffix + '>'
                    
                    sents.append(''.join(tagged_sent))
                
                
                direction_label = 'Rightward' if id1 == subj_id else 'Leftward'
                if subj_id == 'None' or subj_id == '':
                    if re_label != 'None':
                        direction_label = 'No_Direct'
                    else:
                        direction_label = 'None'
                
                if len(considered_ne_pairs) == 1:
                    # is cdr
                    if ne_type1 == 'Chemical':
                        ne1_text, ne2_text = ne2_text, ne1_text

                    instruction = 'Identify the BC5CDR relation label for the highlighted pair, "' + ne1_text + '" and "' + ne2_text + '", and respond in JSON format.'
                    content  = ' '.join(sents)
                    #response = 'For the highlighted pair ' + ne1_text + ' and ' + ne2_text + ':\n' + 'BioRED relation label: ' + re_label + '\n' + 'Direction label: ' + dir_label + '\n' + 'Novelty label: ' + nov_label
                    response = json.dumps({
                        "relation_label": re_label
                    })
                else:
                    instruction = 'Identify the BioRED relation, direction, and novelty labels for the highlighted pair, "' + ne1_text + '" and "' + ne2_text + '", and respond in JSON format.'
                    content  = ' '.join(sents)
                    #response = 'For the highlighted pair ' + ne1_text + ' and ' + ne2_text + ':\n' + 'BioRED relation label: ' + re_label + '\n' + 'Direction label: ' + dir_label + '\n' + 'Novelty label: ' + nov_label
                    response = json.dumps({
                        "BioRED_relation_label": re_label,
                        "direction_label": direction_label,
                        "novelty_label": nov_label
                    })


                messages = []
                messages.append({"role": "system", "content": "You are a bioinformatics expert."})
                messages.append({"role": "user", "content": instruction + "\n" + content})
                messages.append({"role": "assistant", "content": response})
                #messages.append({"prompt": instruction + "\n" + content})
                #messages.append({"completion": response})

                
                json.dump({"messages": messages}, out_writer, ensure_ascii=False)
                #json.dump(messages, out_writer, ensure_ascii=False)
                out_writer.write('\n')
                out_writer.flush()
    out_writer.close()

def dump_documents_2_bioredirect_format(
    all_documents,
    out_bert_file,
    considered_ne_pairs,
    is_biored = True,
    tokenizer = None):
    
    out_writer = open(out_bert_file, 'w', encoding='utf8')

    for document in all_documents:
        pmid = document.id
        
        id_and_ne_types = set()
        for i, text_instance in enumerate(document.text_instances):
            for ann in text_instance.annotations:
                for id in ann.ids:
                    id_and_ne_types.add((id, ann.ne_type))
        #print('=========>len(id_and_ne_types)', len(id_and_ne_types))
        
        for id1, ne_type1 in id_and_ne_types:
            for id2, ne_type2 in id_and_ne_types:
                if ne_type1 > ne_type2:
                    continue
                if ne_type1 == ne_type2 and id1 >= id2:
                    continue
                if id1 == '-1' or id2 == '-1':
                    continue
                if id1 == '-' or id2 == '-':
                    continue
                
                re_label, nov_label, subj_id = 'None', 'None', 'None'
                if (id1, id2) in document.relation_pairs:
                    _tks = document.relation_pairs[(id1, id2)].split('\t')
                    if len(_tks) == 2:
                        re_label, nov_label = _tks
                    elif len(_tks) == 3:
                        re_label, nov_label, subj_id = _tks
                    else:
                        re_label = _tks[0]
                elif (id2, id1) in document.relation_pairs:
                    _tks = document.relation_pairs[(id2, id1)].split('\t')
                    if len(_tks) == 2:
                        re_label, nov_label = _tks
                    elif len(_tks) == 3:
                        re_label, nov_label, subj_id = _tks
                    else:
                        re_label = _tks[0]
                if ((ne_type1, ne_type2) not in considered_ne_pairs) and ((ne_type2, ne_type1) not in considered_ne_pairs):
                    continue

                ''' PMID "\t" + 
                ne_type1 "\t" +
                ne_type2 "\t" +
                ID1 "\t" +
                ID2 "\t" +
                "sent_id|tagged text;.." + "\t"
                "rel_label" + "\t"
                "novelty_label" + "\t"
                '''
                
                if ne_type1 < ne_type2:
                    ne_type1_suffix = 'Src'
                    ne_type2_suffix = 'Tgt'
                elif ne_type1 > ne_type2:
                    ne_type1_suffix = 'Tgt'
                    ne_type2_suffix = 'Src'
                else:
                    ne_type1_suffix = ''
                    for i, text_instance in enumerate(document.text_instances):
                        for ann in text_instance.annotations:
                            ne_type_suffix = ''
                            if id1 in ann.ids:
                                ne_type1_suffix = 'Src'
                                ne_type2_suffix = 'Tgt'
                                break
                            elif id2 in ann.ids:
                                ne_type1_suffix = 'Tgt'
                                ne_type2_suffix = 'Src'
                                break
                        if ne_type1_suffix != '':
                            break
                    
                sents = []
                
                ne1_text = ''
                ne2_text = ''
                ne1_tag  = ''
                ne2_tag  = ''
                for i, text_instance in enumerate(document.text_instances):
                    tagged_sent = list(text_instance.tokens)

                    for ann in text_instance.annotations:
                        _ne_type = ann.ne_type
                        if _ne_type == 'SequenceVariant':
                            _ne_type = 'GeneOrGeneProduct'
                        if id1 in ann.ids:
                            if ne1_text == '':
                                ne1_text = ann.text
                                ne1_tag = _ne_type + ne_type1_suffix
                        elif id2 in ann.ids:
                            if ne2_text == '':
                                ne2_text = ann.text
                                ne2_tag = _ne_type + ne_type2_suffix
                    if ne1_text != '' and ne2_text != '':
                        break
                
                if is_biored:
                    if ne1_tag.endswith('Src'):
                        _prompt = '[CLS] What are the labels of [REL] , [DIR] , and [NOV] between @' + ne1_tag + '$ ' + ne1_text + ' @' + ne1_tag + '/$ and @' + ne2_tag + '$ ' + ne2_text + ' @' + ne2_tag + '/$ ?'
                    else:
                        _prompt = '[CLS] What are the labels of [REL] , [DIR] , and [NOV] between @' + ne2_tag + '$ ' + ne2_text + ' @' + ne2_tag + '/$ and @' + ne1_tag + '$ ' + ne1_text + ' @' + ne1_tag + '/$ ?'
                    if tokenizer != None:
                        _prompt = ' '.join(tokenizer.tokenize(_prompt))
                    sents.append(_prompt)
                else:
                    _prompt = '[CLS] What is the label of [REL] between @' + ne1_tag + '$ ' + ne1_text + ' @' + ne1_tag + '/$ and @' + ne2_tag + '$ ' + ne2_text + ' @' + ne2_tag + '/$ ?'
                    if tokenizer != None:
                        _prompt = ' '.join(tokenizer.tokenize(_prompt))
                    sents.append(_prompt)

                for i, text_instance in enumerate(document.text_instances):
                    tagged_sent = list(text_instance.tokens)

                    for ann in text_instance.annotations:
                        ne_type_suffix = ''
                        if id1 in ann.ids:
                            ne_type_suffix = ne_type1_suffix
                        elif id2 in ann.ids:
                            ne_type_suffix = ne_type2_suffix
                        else:
                            continue

                        _ne_type = ann.ne_type
                        if _ne_type == 'SequenceVariant':
                            _ne_type = 'GeneOrGeneProduct'

                        tagged_sent[ann.start_token] = '@' + _ne_type + ne_type_suffix + '$ ' + tagged_sent[ann.start_token]
                        tagged_sent[ann.end_token] = tagged_sent[ann.end_token] + ' @' + _ne_type + ne_type_suffix + '/$'
                    
                    if i == 0:
                        sents.append('[SEP] ' + ' '.join(tagged_sent))
                    else:
                        sents.append('[SEP] ' + ' '.join(tagged_sent))

            
                if ne_type1 == ne_type2 and ne1_tag.endswith('Tgt'):
                    out_writer.write(pmid + '\t' + \
                                    ne_type1 + '\t' + \
                                    ne_type2 + '\t' + \
                                    id2 + '\t' + \
                                    id1 + '\t' + \
                                    ' '.join(sents) + '\t' + \
                                    re_label + '\t' + \
                                    nov_label + '\t' + \
                                    subj_id + '\n')
                else:
                    out_writer.write(pmid + '\t' + \
                                    ne_type1 + '\t' + \
                                    ne_type2 + '\t' + \
                                    id1 + '\t' + \
                                    id2 + '\t' + \
                                    ' '.join(sents) + '\t' + \
                                    re_label + '\t' + \
                                    nov_label + '\t' + \
                                    subj_id + '\n')
                out_writer.flush()

    out_writer.close()

def dump_documents_2_instruction_format(
    all_documents,
    out_bert_file,
    considered_ne_pairs):
    
    out_writer = open(out_bert_file, 'w', encoding='utf8')
    first_record = True
    for document in all_documents:
        pmid = document.id
        
        id_and_ne_types = set()
        for i, text_instance in enumerate(document.text_instances):
            for ann in text_instance.annotations:
                for id in ann.ids:
                    id_and_ne_types.add((id, ann.ne_type))

        for id1, ne_type1 in id_and_ne_types:
            for id2, ne_type2 in id_and_ne_types:
                if id1 >= id2:
                    continue
                
                #for (id1, id2) in document.relation_pairs:
                #    print(pmid, id1, id2, document.relation_pairs[(id1, id2)])

                re_label, nov_label, subj_id = 'None', 'None', 'None'
                if (id1, id2) in document.relation_pairs:
                    _tks = document.relation_pairs[(id1, id2)].split('\t')
                    if len(_tks) == 1:
                        re_label = _tks[0]
                    elif len(_tks) == 2:
                        re_label, nov_label = _tks
                    elif len(_tks) == 3:
                        re_label, nov_label, subj_id = _tks
                elif (id2, id1) in document.relation_pairs:
                    _tks = document.relation_pairs[(id2, id1)].split('\t')
                    if len(_tks) == 1:
                        re_label = _tks[0]
                    elif len(_tks) == 2:
                        re_label, nov_label = _tks
                    elif len(_tks) == 3:
                        re_label, nov_label, subj_id = _tks
                if ((ne_type1, ne_type2) not in considered_ne_pairs) and ((ne_type2, ne_type1) not in considered_ne_pairs):
                    continue

                ''' PMID "\t" + 
                ne_type1 "\t" +
                ne_type2 "\t" +
                ID1 "\t" +
                ID2 "\t" +
                instruction + "\t"
                context + "\t"
                response + "\t"
                '''
                
                if ne_type1 < ne_type2:
                    ne_type1_suffix = 'Src'
                    ne_type2_suffix = 'Tgt'
                elif ne_type1 > ne_type2:
                    ne_type1_suffix = 'Tgt'
                    ne_type2_suffix = 'Src'
                else:
                    ne_type1_suffix = ''
                    for i, text_instance in enumerate(document.text_instances):
                        for ann in text_instance.annotations:
                            ne_type_suffix = ''
                            if id1 in ann.ids:
                                ne_type1_suffix = 'Src'
                                ne_type2_suffix = 'Tgt'
                                break
                            elif id2 in ann.ids:
                                ne_type1_suffix = 'Tgt'
                                ne_type2_suffix = 'Src'
                                break
                        if ne_type1_suffix != '':
                            break

                if len(considered_ne_pairs) == 1:
                    ne_type1_suffix = ''
                    ne_type2_suffix = ''
                    
                sents = []
                ne1_text = ''
                ne2_text = ''
                ne1_tag = ''
                ne2_tag = ''
                for i, text_instance in enumerate(document.text_instances):
                    tagged_sent = [str(c) for c in text_instance.text]

                    for ann in text_instance.annotations:
                        ne_type_suffix = ''
                        if id1 in ann.ids:
                            ne_type_suffix = ne_type1_suffix
                            if ne1_text == '':
                                ne1_text = ann.text
                        elif id2 in ann.ids:
                            ne_type_suffix = ne_type2_suffix
                            if ne2_text == '':
                                ne2_text = ann.text
                        else:
                            continue

                        _ne_type = ann.ne_type
                        if _ne_type == 'SequenceVariant':
                            _ne_type = 'GeneOrGeneProduct'

                        tagged_sent[ann.position] = '<' + _ne_type + ne_type_suffix + '> ' + tagged_sent[ann.position]
                        tagged_sent[ann.position + ann.length - 1] = tagged_sent[ann.position + ann.length - 1] + ' </' + _ne_type + ne_type_suffix + '>'
                    
                    sents.append(''.join(tagged_sent))
                
                
                direction_label = 'Rightward' if id1 == subj_id else 'Leftward'
                if subj_id == 'None' or subj_id == '':
                    if re_label != 'None':
                        direction_label = 'No_Direct'
                    else:
                        direction_label = 'None'
                
                if len(considered_ne_pairs) == 1:
                    # Dataset is BC5CDR
                    if ne_type1 == 'Chemical':
                        ne1_text, ne2_text = ne2_text, ne1_text

                    instruction = 'Identify the BC5CDR relation label for the highlighted pair, "' + ne1_text + '" and "' + ne2_text + '", and respond in JSON format.'
                    content  = re.sub(r'\s+', ' ', ' '.join(sents))
                    #response = 'For the highlighted pair ' + ne1_text + ' and ' + ne2_text + ':\n' + 'BioRED relation label: ' + re_label + '\n' + 'Direction label: ' + dir_label + '\n' + 'Novelty label: ' + nov_label
                    response = json.dumps({
                        "relation_label": re_label
                    })
                else:
                    instruction = 'Identify the BioRED relation, direction, and novelty labels for the highlighted pair, "' + ne1_text + '" and "' + ne2_text + '", and respond in JSON format.'
                    content  = re.sub(r'\s+', ' ', ' '.join(sents))
                    #response = 'For the highlighted pair ' + ne1_text + ' and ' + ne2_text + ':\n' + 'BioRED relation label: ' + re_label + '\n' + 'Direction label: ' + dir_label + '\n' + 'Novelty label: ' + nov_label
                    response = json.dumps({
                        "BioRED_relation_label": re_label,
                        "direction_label": direction_label,
                        "novelty_label": nov_label
                    })


                messages = []
                messages.append({"role": "system", "content": "You are a bioinformatics expert."})
                messages.append({"role": "user", "content": instruction + "\n" + content})
                messages.append({"role": "assistant", "content": response})
                #messages.append({"prompt": instruction + "\n" + content})
                #messages.append({"completion": response})

                
                json.dump({"messages": messages}, out_writer, ensure_ascii=False)
                #json.dump(messages, out_writer, ensure_ascii=False)
                out_writer.write('\n')
                out_writer.flush()
    out_writer.close()