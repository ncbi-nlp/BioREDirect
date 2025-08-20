import argparse
import glob
import logging
import numpy as np
import os
import random
import torch

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from transformers import set_seed, BertTokenizer
from torch.utils.data import DataLoader

from data_processor import BioREDDataset
from dataset_format_converter.convert_bioc_2_tsv import load_bioc_into_documents
from dataset_format_converter.convert_bioc_2_pubtator3 import dump_documents_2_xml_with_instance
from dataset_format_converter.convert_pubtator_2_tsv import load_pubtator_into_documents
from dataset_format_converter.convert_pubtator_2_pubtator3 import dump_documents_2_pubtator3
from dataset_format_converter.utils import split_documents, dump_documents_2_bioredirect_format
from evaluation import convert_to_biored_label_with_score
from models import BioREDirect

# Set up logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RelInfo:
    
    def __init__(self, 
                 id1, id2, 
                 rel_type, novelty_type, direction_type,
                 rel_score, novelty_score, direction_score):
        self.id1   = id1
        self.id2   = id2
        self.rel_type  = rel_type
        self.rel_score = rel_score
        self.novelty_type  = novelty_type
        self.novelty_score = novelty_score
        self.direction_type  = direction_type
        self.direction_score = direction_score

def custom_collate_fn(batch):
    pmid             = [item['pmid'] for item in batch]
    input_ids        = torch.stack([item['input_ids'] for item in batch])
    attention_mask   = torch.stack([item['attention_mask'] for item in batch])
    relation_labels  = torch.stack([item['relation_labels'] for item in batch])
    novelty_labels   = torch.stack([item['novelty_labels'] for item in batch])
    direction_labels = torch.stack([item['direction_labels'] for item in batch])
    pair_prompt_ids  = torch.stack([item['pair_prompt_ids'] for item in batch])
    relation_token_index  = torch.stack([item['relation_token_index'] for item in batch])
    direction_token_index = torch.stack([item['direction_token_index'] for item in batch])
    novelty_token_index   = torch.stack([item['novelty_token_index'] for item in batch])

    return {
        "pmid":                  pmid,
        "input_ids":             input_ids,
        "attention_mask":        attention_mask,
        "relation_labels":       relation_labels,
        "novelty_labels":        novelty_labels,
        "direction_labels":      direction_labels,
        "pair_prompt_ids":       pair_prompt_ids,
        "relation_token_index":  relation_token_index,
        "direction_token_index": direction_token_index,
        "novelty_token_index":   novelty_token_index,
    }

def load_tokenizer(in_bert_model, dataset_processor):
    tokenizer = BertTokenizer.from_pretrained(in_bert_model)
    to_add_special_tokens = set()
    merged_vocab = tokenizer.get_vocab()
    for token in dataset_processor.get_special_tokens() + ['[REL]', '[DIR]', '[NOV]']:
        if (token not in merged_vocab) and (token not in tokenizer.additional_special_tokens):
            to_add_special_tokens.add(token)
    to_add_special_tokens = list(to_add_special_tokens)
    to_add_special_tokens.sort()
    tokenizer.add_special_tokens({'additional_special_tokens': tokenizer.additional_special_tokens + to_add_special_tokens})
    return tokenizer

def run_inference(bioredirect_model,
                  tokenizer,
                  test_dataset,
                  test_dataloader,
                  device):
    
    relation_label_list  = test_dataset.get_labels('relation')
    novelty_label_list   = test_dataset.get_labels('novelty')
    direction_label_list = test_dataset.get_labels('direction')
    
    bioredirect_model.eval()
    
    pred_infos = []
    with torch.no_grad():
        total_batches = len(test_dataloader)
        for i, batch in tqdm(enumerate(test_dataloader), total=total_batches, desc="Evaluation"):
            
            (rel_token_outputs, 
             nov_token_outputs, 
             dir_token_outputs) = bioredirect_model(input_ids             = batch['input_ids'].to(device), 
                                                    attention_mask        = batch['attention_mask'].to(device), 
                                                    pair_prompt_ids       = batch['pair_prompt_ids'].to(device), 
                                                    relation_token_index  = batch['relation_token_index'],
                                                    direction_token_index = batch['direction_token_index'],
                                                    novelty_token_index   = batch['novelty_token_index'])

            pred_rel_scores = torch.sigmoid(rel_token_outputs).cpu().numpy()
            pred_nov_scores = torch.sigmoid(nov_token_outputs).cpu().numpy()
            pred_dir_scores = torch.sigmoid(dir_token_outputs).cpu().numpy()

            for (rel_label_idx, nov_label_idx, dir_label_idx,
                 rel_label_score, nov_label_score, dir_label_score) in convert_to_biored_label_with_score(pred_rel_scores, 
                                                                                                          pred_nov_scores,
                                                                                                          pred_dir_scores,
                                                                                                          relation_label_list,
                                                                                                          novelty_label_list,
                                                                                                          direction_label_list):
                pred_infos.append((relation_label_list[rel_label_idx],
                                   novelty_label_list[nov_label_idx],
                                   direction_label_list[dir_label_idx],
                                   rel_label_score,
                                   nov_label_score,
                                   dir_label_score))
    
    pmid_2_rel_pairs_dict  = defaultdict(list)

    for (_, data), pred_info in zip(test_dataset.data.iterrows(), pred_infos):
        
        pmid = str(data[0])
        id1  = str(data[3])
        id2  = str(data[4])
        if pred_info[0] == 'None':
            continue
        pmid_2_rel_pairs_dict[pmid].append(RelInfo(id1, id2, 
                                                   pred_info[0], pred_info[1], pred_info[2],
                                                   pred_info[3], pred_info[4], pred_info[5]))

    return pmid_2_rel_pairs_dict
    
def run_inference_dir(in_bioredirect_model,
                      format,
                      in_data_dir,
                      out_pred_data_dir,
                      re_id_spliter_str = r'[\;]',
                      normalized_ne_type_dict = {},
                      sections = '',
                      batch_size  = 16,
                      max_seq_len = 512,
                      use_single_chunk = False):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
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
    
    tokenizer = BertTokenizer.from_pretrained(in_bioredirect_model)    
    bioredirect_model = BioREDirect.load_model(model_path = in_bioredirect_model, use_single_chunk = use_single_chunk)
    
    _in_files = list(glob.glob(in_data_dir + "/*.xml") + glob.glob(in_data_dir + "/*.txt") + glob.glob(in_data_dir + "/*.pubtator"))
    random.seed(datetime.now().timestamp())
    random.shuffle(_in_files)

    print(f'Found {_in_files} files in {in_data_dir}')
    logger.info(f'Found {_in_files} files in {in_data_dir}')

    len_in_files = len(_in_files)
    counter = 0

    try:
        if not os.path.exists(in_data_dir + '_processed'):
            os.makedirs(in_data_dir + '_processed')
    except Exception as e:
        if not os.path.exists(in_data_dir + '_processed'):
            print(f'Error creating directory {in_data_dir + "_processed"}: {e}')

    try:
        if not os.path.exists(out_pred_data_dir):
            os.makedirs(out_pred_data_dir)
    except Exception as e:
        if not os.path.exists(out_pred_data_dir):
            print(f'Error creating directory {out_pred_data_dir}: {e}')

    for in_test_data_file in _in_files:
        
        file_name          = Path(in_test_data_file).stem
        in_data_file       = in_data_dir + '//' + Path(in_test_data_file).name
        in_test_tsv_file   = in_data_dir + '_processed//' + file_name + '.tsv'
        out_pred_data_file = out_pred_data_dir + '//' + Path(in_test_data_file).name
        
        
        if os.path.exists(out_pred_data_file):
            continue
        
        if format.lower() == 'bioc':
            documents = load_bioc_into_documents(
                in_bioc_xml_file        = in_data_file, 
                re_id_spliter_str       = re_id_spliter_str,
                normalized_ne_type_dict = normalized_ne_type_dict,
                sections                = sections)
        else:
            documents = load_pubtator_into_documents(
                in_pubtator_file        = in_data_file, 
                re_id_spliter_str       = re_id_spliter_str,
                normalized_ne_type_dict = normalized_ne_type_dict,
                use_novelty_label       = True)

        split_documents(documents, tokenizer)

        dump_documents_2_bioredirect_format(
            all_documents       = documents, 
            out_bert_file       = in_test_tsv_file,
            considered_ne_pairs = considered_ne_pairs,
            tokenizer           = tokenizer)

        if os.path.getsize(in_test_tsv_file) == 0:
            if not os.path.exists(in_test_tsv_file):
                with open(in_test_tsv_file, "w") as writer:
                    print(in_test_tsv_file, ' is empty')
            if not os.path.exists(out_pred_data_file):
                # copy the input file to output directory
                os.system(f'cp {in_data_file} {out_pred_data_file}')
            counter += 1
            continue

        if not os.path.exists(out_pred_data_file):
            
            test_dataset    = BioREDDataset(in_test_tsv_file,
                                            tokenizer,
                                            max_seq_len     = max_seq_len,
                                            soft_prompt_len = bioredirect_model.soft_prompt_len,
                                            use_single_chunk = use_single_chunk,)
            
            test_dataloader = DataLoader(test_dataset, 
                                         batch_size = batch_size, 
                                         collate_fn = custom_collate_fn,
                                         num_workers = 10, 
                                         pin_memory = (device.type == 'cuda'))
            
            pmid_2_rel_pairs_dict = run_inference(bioredirect_model = bioredirect_model,
                                                  tokenizer         = tokenizer,
                                                  test_dataset      = test_dataset,
                                                  test_dataloader   = test_dataloader,
                                                  device            = device)
            
            if format.lower() == 'bioc':
                dump_documents_2_xml_with_instance(
                        in_xml_file  = in_data_file,
                        documents    = documents, 
                        out_xml_file = out_pred_data_file,
                        pmid_2_rel_pair_dict = pmid_2_rel_pairs_dict)
            else:
                dump_documents_2_pubtator3(
                        in_data_file         = in_data_file,
                        out_data_file        = out_pred_data_file,
                        pmid_2_rel_pair_dict = pmid_2_rel_pairs_dict,
                        re_id_spliter_str    = re_id_spliter_str)
        counter += 1
        if counter % 10 == 0:
            print(f'Processed {counter}/{len_in_files} {counter/len_in_files*100:.2f}% files')
            logger.info(f'Processed {counter}/{len_in_files} {counter/len_in_files*100:.2f}% files')
            
if __name__ == '__main__':

    torch.set_default_dtype(torch.float32)
    
    parser = argparse.ArgumentParser(description='Run Relation Extraction Experiment')
    parser.add_argument('--in_bioredirect_model', type=str, default='',  help='Output BERT model name')
    parser.add_argument('--in_data_dir',          type=str, default='',  help='Input BIOC XML directory')
    parser.add_argument('--out_pred_data_dir',    type=str, default='',  help='Output BIOC XML directory')
    parser.add_argument('--batch_size',           type=int, default=8,   help='Batch size')
    parser.add_argument('--max_seq_len',          type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--sections',             type=str, default='',  help='Sections to process separated by |')
    parser.add_argument('--format',               type=str, default='bioc', choices=['bioc', 'pubtator'], help='Dataset format')
    parser.add_argument('--use_single_chunk',     type=bool, default=False, help='Use single chunk for training')
    args = parser.parse_args() 

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

    run_inference_dir(in_bioredirect_model    = args.in_bioredirect_model,
                      format                  = args.format,
                      in_data_dir             = args.in_data_dir,
                      out_pred_data_dir       = args.out_pred_data_dir,
                      normalized_ne_type_dict = normalized_ne_type_dict,
                      re_id_spliter_str       = r'[\;]',
                      max_seq_len             = args.max_seq_len,
                      batch_size              = args.batch_size,
                      sections                = args.sections,
                      use_single_chunk        = args.use_single_chunk)
