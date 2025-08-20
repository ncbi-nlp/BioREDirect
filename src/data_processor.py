from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import pandas as pd

# Set up logger
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BioREDDataset(Dataset):

    def __init__(self, 
                 filename, 
                 tokenizer, 
                 max_seq_len=512,
                 soft_prompt_len=10,
                 text_col=5,
                 label_col=-3,
                 subject_id_col=-1,
                 ne_id1_col=3,
                 ne_id2_col=4,
                 pair_cols=[1, 2],
                 novelty_label_col=-2,
                 balance_ratio=-1,
                 use_single_chunk=False):

        self.data = pd.read_csv(filename,
                                sep='\t', 
                                header=None,
                                dtype=str).fillna(np.str_('None'))
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.text_col = text_col
        self.label_col = label_col
        self.subject_id_col = subject_id_col
        self.novelty_label_col = novelty_label_col
        self.ne_id1_col = ne_id1_col
        self.ne_id2_col = ne_id2_col
        self.soft_prompt_len = soft_prompt_len
        self.pair_cols = pair_cols
        self.considered_ne_pairs = [('ChemicalEntity', 'ChemicalEntity'),
                                    ('ChemicalEntity', 'DiseaseOrPhenotypicFeature'),
                                    ('ChemicalEntity', 'GeneOrGeneProduct'),
                                    ('ChemicalEntity', 'SequenceVariant'),
                                    ('DiseaseOrPhenotypicFeature', 'GeneOrGeneProduct'),
                                    ('DiseaseOrPhenotypicFeature', 'SequenceVariant'),
                                    ('GeneOrGeneProduct', 'GeneOrGeneProduct'),
                                    ('SequenceVariant', 'SequenceVariant')]
        self.ne_type_pair_2_id = {ne_type_pair: i for i, ne_type_pair in enumerate(self.considered_ne_pairs)}        
        self.use_single_chunk = use_single_chunk
        
        if balance_ratio > 0:
            none_data = self.data[self.data.iloc[:, label_col] == 'None']
            non_none_data = self.data[self.data.iloc[:, label_col] != 'None']
            num_non_none = non_none_data.shape[0]
            num_none_to_keep = balance_ratio * num_non_none  # twice the number of non-'None' instances
            none_data = none_data.sample(n=num_none_to_keep, random_state=1111)
            # Combine the datasets back
            combined_data = pd.concat([non_none_data, none_data]).reset_index(drop=True)
            # Shuffle the combined data
            self.data = combined_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    def __get_ne_type_pair__(self, text):

        # text: "What is [Litcoin] between @GeneOrGeneProductSrc$ Na(v)1.5 @/GeneOrGeneProductSrc$ and @GeneOrGeneProductTgt$ G-->A substitution at codon 1763 @/GeneOrGeneProductTgt$ ?"
        # entity_pair: ['GeneOrGeneProduct', 'GeneOrGeneProduct']
        ne_type1 = ''
        ne_type2 = ''
        for token in text.split():
            if token.startswith('@') and token.endswith('Src$'):
                ne_type1 = token[1:-4]
            elif token.startswith('@') and token.endswith('Tgt$'):
                ne_type2 = token[1:-4]
            if ne_type1 != '' and ne_type2 != '':
                break
        if ne_type2 < ne_type1:
            ne_type1, ne_type2 = ne_type2, ne_type1
        #logger.info(text)
        return (ne_type1, ne_type2)
    
    def __len__(self):
        return len(self.data)
    
    def __get_prefix_and_suffix_chunks__(self, 
                       text_content, 
                       max_content_size=512):
        num_words = len(text_content["text_tokens"])
        prefix_chunk = None
        suffix_chunk = None
        
        # Calculate the number of chunks needed
        # get the ceiling value of the division
        num_chunks = np.ceil(num_words / max_content_size)
        
        if num_chunks >= 1:
            start_idx = 0 * (max_content_size - 1)
            end_idx = start_idx + min(max_content_size, num_words)
            prefix_chunk = text_content["text_tokens"][start_idx:end_idx]
            suffix_chunk = text_content["text_tokens"][start_idx:end_idx]
        if num_chunks > 1:
            start_idx = num_words - max_content_size
            end_idx = num_words
            suffix_chunk = text_content["text_tokens"][start_idx:end_idx]
        
        return prefix_chunk, suffix_chunk
    
    def __get_closest_chunks__(self, text_content, max_content_size=512):

        sent_chunks = []
        out_chunks = []

        src_sent_ids = []
        tgt_sent_ids = []

        current_chunk = []
        sent_id = 0
        for i, token in enumerate(text_content["text_tokens"]):
            if token == '[SEP]' and i != 0:
                # check if the current chunk contains the entity pair
                has_src, has_tgt = False, False
                for t in current_chunk:
                    if t.startswith('@') and t.endswith('Src$'):
                        has_src = True
                        if sent_id not in src_sent_ids:
                            src_sent_ids.append(sent_id)
                    elif t.startswith('@') and t.endswith('Tgt$'):
                        has_tgt = True
                        if sent_id not in tgt_sent_ids:
                            tgt_sent_ids.append(sent_id)
                    if has_src and has_tgt:
                        break
                sent_chunks.append(current_chunk)
                current_chunk = []
                current_chunk.append(token)
                sent_id += 1
            else:
                current_chunk.append(token)
                
        if len(current_chunk) > 1:
            has_src, has_tgt = False, False
            for t in current_chunk:
                if t.startswith('@') and t.endswith('Src$'):
                    has_src = True
                    if sent_id not in src_sent_ids:
                        src_sent_ids.append(sent_id)
                elif t.startswith('@') and t.endswith('Tgt$'):
                    has_tgt = True
                    if sent_id not in tgt_sent_ids:
                        tgt_sent_ids.append(sent_id)
                if has_src and has_tgt:
                    break
            sent_chunks.append(current_chunk)

        min_distance = 1000000
        for src_sent_id in src_sent_ids:
            for tgt_sent_id in tgt_sent_ids:
                if abs(src_sent_id - tgt_sent_id) < min_distance:
                    min_distance = abs(src_sent_id - tgt_sent_id)
        duplicated_chunks = set()
        for src_sent_id in src_sent_ids:
            for tgt_sent_id in tgt_sent_ids:
                if abs(src_sent_id - tgt_sent_id) <= min_distance and ((src_sent_id, tgt_sent_id) not in duplicated_chunks):
                    chunk = []
                    for i in range(min(src_sent_id, tgt_sent_id), max(src_sent_id, tgt_sent_id) + 1):
                        chunk += sent_chunks[i]
                    
                    if len(out_chunks) > 0 and (len(out_chunks[-1]) + len(chunk) < max_content_size):
                        out_chunks[-1] += chunk
                    elif len(chunk) < max_content_size:
                        out_chunks.append(chunk)

                    duplicated_chunks.add((src_sent_id, tgt_sent_id))
                    duplicated_chunks.add((tgt_sent_id, src_sent_id))
        
        #print('=================>out_chunks[0]', out_chunks[0])
        return out_chunks

    def __get_input_ids__(self, 
                          tokens, 
                          soft_prompt_len, 
                          padding_length):

        attention_mask = [1] * (len(tokens) + soft_prompt_len)
        attention_mask += [0] * padding_length

        #print('=================>len(window_tokens)', len(window_tokens), window_tokens)

        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        token_ids += [self.tokenizer.pad_token_id] * padding_length
        entity1_indices = []
        entity2_indices = []
        entity1_sent_ids = []
        entity2_sent_ids = []
        current_entity_indices = []
        begin_entity1 = False
        begin_entity2 = False
        sent_id = 0
        sent_ids = []
        for j, token in enumerate(tokens):
            if token == '[SEP]':
                sent_id += 1
            if token == '[CLS]' or token == '[SEP]':
                sent_ids.append(sent_id)
            # end of entity1 token
            #print(token)
            if token.startswith('@') and token.endswith('Src/$'):
                current_entity_indices.append(j)
                entity1_indices.append(current_entity_indices)
                entity1_sent_ids.append(sent_id)
                current_entity_indices = []
                begin_entity1 = False
                begin_entity2 = False
            elif token.startswith('@') and token.endswith('Src$'):
                # start of entity1 token
                current_entity_indices.append(j)
                begin_entity1 = True
            elif token.startswith('@') and token.endswith('Tgt/$'):
                # end of entity2 token
                current_entity_indices.append(j)
                entity2_indices.append(current_entity_indices)
                entity2_sent_ids.append(sent_id)
                current_entity_indices = []
                begin_entity2 = False
                begin_entity2 = False
            elif token.startswith('@') and token.endswith('Tgt$'):
                # start of entity2 token
                current_entity_indices.append(j)
                begin_entity2 = True
        
        return {'input_ids': token_ids,
                'attention_mask': attention_mask,
                'entity1_indices': entity1_indices,
                'entity2_indices': entity2_indices,
                'entity1_sent_ids': entity1_sent_ids,
                'entity2_sent_ids': entity2_sent_ids,
                'sent_ids': sent_ids}

    def __add_chunks_to_inputs__(self, 
                                 all_inputs, 
                                 chunks, 
                                 question_prompt_tokens,
                                 entity_pair):

        len_question_prompt_tokens = len(question_prompt_tokens)

        for i, window_tokens in enumerate(chunks):
            padding_length = self.max_seq_len - (self.soft_prompt_len + len_question_prompt_tokens + len(window_tokens) + 1)
            
            _window_tokens = question_prompt_tokens + ['[SEP]'] + window_tokens

            inputs = self.__get_input_ids__(_window_tokens, self.soft_prompt_len, padding_length)
            inputs['pair_prompt_ids'] = self.ne_type_pair_2_id[entity_pair] if entity_pair in self.ne_type_pair_2_id else len(self.ne_type_pair_2_id)

            if len(inputs['input_ids']) + self.soft_prompt_len!= self.max_seq_len:
                print(len(inputs['input_ids']) + self.soft_prompt_len)
                print(window_tokens)

            all_inputs['input_ids'].append(inputs['input_ids'])
            all_inputs['attention_mask'].append(inputs['attention_mask'])
            #all_inputs['entity1_indices'].append(inputs['entity1_indices'])
            #all_inputs['entity2_indices'].append(inputs['entity2_indices'])
            #all_inputs['entity1_sent_ids'].append(inputs['entity1_sent_ids'])
            #all_inputs['entity2_sent_ids'].append(inputs['entity2_sent_ids'])
            all_inputs['pair_prompt_ids'].append(inputs['pair_prompt_ids'])
            #all_inputs['sent_ids'].append(inputs['sent_ids'])

    def __getitem__(self, idx):

        #text = self.data.iloc[idx, self.text_col].split(' [SEP] ', 1)[-1]
        #print(self.data.iloc[idx, self.text_col])
        #soft_prompt_id = self.ne_type_pair_2_id[entity_pair]
        pmid        = str(self.data.iloc[idx, 0])
        tokens      = self.data.iloc[idx, self.text_col].split(' ')
        #entity_pair = self.__get_ne_type_pair__(self.data.iloc[idx, self.text_col])
        ne_type1 = self.data.iloc[idx, self.pair_cols[0]]
        ne_type2 = self.data.iloc[idx, self.pair_cols[1]]
        entity_pair = (ne_type1, ne_type2) if ne_type1 <= ne_type2 else (ne_type2, ne_type1)
        
        all_inputs = {
            "input_ids": [],
            "attention_mask": [],
            #"entity1_indices": [],
            #"entity2_indices": [],
            #"entity1_sent_ids": [],
            #"entity2_sent_ids": [],
            "pair_prompt_ids": [],
            #"sent_ids": []
        }

        content = {
            "instruction_tokens": [],
            "text_tokens": [],
        }
        
        relation_token_index = 0
        direction_token_index = 0
        novelty_token_index = 0
        instruction_end = 0
        for i, token in enumerate(tokens):
            if token == '[REL]':
                relation_token_index = i
            elif token == '[DIR]':
                direction_token_index = i
            elif token == '[NOV]':
                novelty_token_index = i
            elif token == '[SEP]':
                break
            content["instruction_tokens"].append(token)
        num_instruction_tokens = len(content["instruction_tokens"])
        content["text_tokens"] = tokens[instruction_end+1:]

        max_content_size = self.max_seq_len - (self.soft_prompt_len + num_instruction_tokens + 1) # 1 for [SEP]

        '''
        segment text into chunks if text is longer than max_length
        '''
        if self.use_single_chunk:
            prefix_chunk, suffix_chunk = self.__get_prefix_and_suffix_chunks__(content, max_content_size)
            if len(content["text_tokens"]) >= max_content_size:
                infix_chunks = self.__get_closest_chunks__(content, max_content_size)
                if len(infix_chunks) == 0:
                    infix_chunks.append(suffix_chunk)
                self.__add_chunks_to_inputs__(all_inputs, [infix_chunks[0]], content["instruction_tokens"], entity_pair) # Maintain Biorex2's effectiveness by dealing with the first chunk
            else:
                self.__add_chunks_to_inputs__(all_inputs, [prefix_chunk], content["instruction_tokens"], entity_pair)
        else:
            prefix_chunk, suffix_chunk = self.__get_prefix_and_suffix_chunks__(content, max_content_size)
            infix_chunks               = self.__get_closest_chunks__(content, max_content_size)
            if len(infix_chunks) == 0:
                infix_chunks.append(suffix_chunk)
            self.__add_chunks_to_inputs__(all_inputs, [prefix_chunk], content["instruction_tokens"], entity_pair)
            self.__add_chunks_to_inputs__(all_inputs, [suffix_chunk], content["instruction_tokens"], entity_pair)
            self.__add_chunks_to_inputs__(all_inputs, [infix_chunks[0]],  content["instruction_tokens"], entity_pair) # Maintain Biorex2's effectiveness by dealing with the first chunk

        label = self.data.iloc[idx, self.label_col]
        #print('============>', self.label_col)
        relation_label_ids = self.get_multi_label_ids(label, 'relation')  # Convert label text to binary array
        relation_label_ids = [int(a) for a in relation_label_ids]
        
        novelty_label = self.data.iloc[idx, self.novelty_label_col]
        novelty_label_ids = self.get_multi_label_ids(novelty_label, 'novelty')  # Convert label text to binary array
        novelty_label_ids = [int(b) for b in novelty_label_ids]

        subject_id = self.data.iloc[idx, self.subject_id_col]
        subject_label_ids = self.get_multi_label_ids('None', 'direction')
        if label != 'None':
            if subject_id == 'None':
                subject_label_ids = self.get_multi_label_ids('No_Direct', 'direction')
            elif subject_id == self.data.iloc[idx, self.ne_id1_col]:
                subject_label_ids = self.get_multi_label_ids('Left_to_Right', 'direction')
            elif subject_id == self.data.iloc[idx, self.ne_id2_col]:
                subject_label_ids = self.get_multi_label_ids('Right_to_Left', 'direction')
        subject_label_ids = [int(b) for b in subject_label_ids]
        #print(subject_label_ids)
        # Perform the bitwise OR operation
        label_ids = [a | b for a, b in zip(relation_label_ids, novelty_label_ids)]
        label_ids = [a | b for a, b in zip(label_ids, subject_label_ids)]
        
        #all_token_ids = all_token_ids
        #all_attention_masks = all_attention_masks
        #all_entity1_indices = all_entity1_indices
        #all_entity2_indices = all_entity2_indices
        #all_entity1_sent_ids = all_entity1_sent_ids
        #all_entity2_sent_ids = all_entity2_sent_ids
        #all_pair_prompt_ids = all_pair_prompt_ids
        return {"pmid": pmid,
                "input_ids": torch.tensor(all_inputs['input_ids']), 
                "attention_mask": torch.tensor(all_inputs['attention_mask']),
                "labels": torch.tensor(label_ids, dtype=torch.float),
                "relation_labels": torch.tensor(relation_label_ids, dtype=torch.float),
                "novelty_labels": torch.tensor(novelty_label_ids, dtype=torch.float),
                "direction_labels": torch.tensor(subject_label_ids, dtype=torch.float),
                "relation_token_index": torch.tensor(relation_token_index, dtype=torch.long),
                "direction_token_index": torch.tensor(direction_token_index, dtype=torch.long),
                "novelty_token_index": torch.tensor(novelty_token_index, dtype=torch.long),
                #"entity1_indices": all_inputs['entity1_indices'],
                #"entity2_indices": all_inputs['entity2_indices'],
                #"entity1_sent_ids": all_inputs['entity1_sent_ids'],
                #"entity2_sent_ids": all_inputs['entity2_sent_ids'],
                "pair_prompt_ids": torch.tensor(all_inputs['pair_prompt_ids']),
                #"sent_ids": all_inputs['sent_ids']
                }
    
    def get_label_2_id(self, task):
        return {label: i for i, label in enumerate(self.get_labels(task))}
        
    def get_labels(self, task):

        if task == 'relation':
            return ['None',
                    'Association',
                    'Bind',
                    'Comparison',
                    'Conversion',
                    'Cotreatment',
                    'Drug_Interaction',
                    'Negative_Correlation',
                    'Positive_Correlation']
        elif task == 'novelty':
            return ['None', 'Novel', 'No']
        elif task == 'direction':
            return ['None', 'Left_to_Right', 'Right_to_Left', 'No_Direct']
        else:
            return ['None']
        
    def get_multi_label_ids(self, label_text, task):
        num_labels = len(self.get_label_2_id(task))
        # Initialize label array with zeros
        labels_array = np.zeros(num_labels)
        
        # Set the index for the primary label
        if num_labels > 1:
            primary_label_index = self.get_label_2_id(task)[label_text]
            labels_array[primary_label_index] = 1
        
        # Check for additional association label
        #if label_text in ['Negative_Correlation', 'Positive_Correlation']:
        #    association_index = self.get_label_2_id(task)['Association']
        #    labels_array[association_index] = 1
        
        return labels_array
    
    @classmethod
    def get_special_tokens(self):
        # Define new special tokens
        return ['@/ChemicalEntitySrc$', '@/ChemicalEntityTgt$', '@/DiseaseOrPhenotypicFeatureSrc$', '@/DiseaseOrPhenotypicFeatureTgt$', '@/GeneOrGeneProductSrc$', '@/GeneOrGeneProductTgt$', '@ChemicalEntitySrc$', '@ChemicalEntitySrc/$', '@ChemicalEntityTgt$', '@ChemicalEntityTgt/$', '@DiseaseOrPhenotypicFeatureSrc$', '@DiseaseOrPhenotypicFeatureSrc/$', '@DiseaseOrPhenotypicFeatureTgt$', '@DiseaseOrPhenotypicFeatureTgt/$', '@GeneOrGeneProductSrc$', '@GeneOrGeneProductSrc/$', '@GeneOrGeneProductTgt$', '@GeneOrGeneProductTgt/$', '[Litcoin]']

class CDRDataset(BioREDDataset):

    def __init__(self, 
                 filename, 
                 tokenizer, 
                 max_seq_len=512,
                 soft_prompt_len=10,
                 text_col=5,
                 label_col=-3,
                 subject_id_col=-1,
                 ne_id1_col=3,
                 ne_id2_col=4,
                 pair_cols=[1, 2],
                 novelty_label_col=-2,
                 balance_ratio=-1):
        
        super().__init__(filename  = filename, 
                         tokenizer = tokenizer, 
                         max_seq_len = max_seq_len,
                         soft_prompt_len = soft_prompt_len,
                         text_col = text_col,
                         label_col = label_col,
                         subject_id_col = subject_id_col,
                         ne_id1_col = ne_id1_col,
                         ne_id2_col = ne_id2_col,
                         pair_cols = pair_cols,
                         novelty_label_col = novelty_label_col,
                         balance_ratio = balance_ratio)
        
    def get_labels(self, task):

        if task == 'relation':
            return ['None', 'CID']
        else:
            return ['None']
        
    @classmethod
    def get_special_tokens(self):
        # Define new special tokens
        return ['@/ChemicalSrc$', '@/ChemicalTgt$', '@/DiseaseSrc$', '@/DiseasTgt$', '@ChemicalSrc$', '@ChemicalSrc/$', '@ChemicalTgt$', '@ChemicalTgt/$', '@DiseaseSrc$', '@DiseaseSrc/$', '@DiseaseTgt$', '@DiseaseTgt/$', '[REL]']

