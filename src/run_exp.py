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

# Set up logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def custom_collate_fn(batch):
    pmid = [item['pmid'] for item in batch]
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    #labels = torch.stack([item['labels'] for item in batch])
    relation_labels = torch.stack([item['relation_labels'] for item in batch])
    novelty_labels = torch.stack([item['novelty_labels'] for item in batch])
    direction_labels = torch.stack([item['direction_labels'] for item in batch])
    #entity1_indices = [item['entity1_indices'] for item in batch]
    #entity2_indices = [item['entity2_indices'] for item in batch]
    #entity1_sent_ids = [item['entity1_sent_ids'] for item in batch]
    #entity2_sent_ids = [item['entity2_sent_ids'] for item in batch]
    pair_prompt_ids = torch.stack([item['pair_prompt_ids'] for item in batch])
    #sent_ids        = [item['sent_ids'] for item in batch]
    relation_token_index =  torch.stack([item['relation_token_index'] for item in batch])
    direction_token_index =  torch.stack([item['direction_token_index'] for item in batch])
    novelty_token_index =  torch.stack([item['novelty_token_index'] for item in batch])
    #print('===============>len(entity2_sent_ids)', len(entity2_sent_ids))

    return {
        "pmid": pmid,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        #"labels": labels,
        "relation_labels": relation_labels,
        "novelty_labels": novelty_labels,
        "direction_labels": direction_labels,
        #"entity1_indices": entity1_indices,
        #"entity2_indices": entity2_indices,
        #"entity1_sent_ids": entity1_sent_ids,
        #"entity2_sent_ids": entity2_sent_ids,
        "pair_prompt_ids": pair_prompt_ids,
        #"sent_ids": sent_ids,        
        "relation_token_index": relation_token_index,
        "direction_token_index": direction_token_index,
        "novelty_token_index": novelty_token_index,
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

def run_train(in_bert_model, 
              in_train_tsv_file,
              in_dev_tsv_file,
              out_bioredirect_model, 
              soft_prompt_len  = 5,
              num_epochs       = 10,
              batch_size       = 16,
              max_seq_len      = 512,
              neg_loss_weight  = 0.5,
              learning_rate    = 1e-5,
              task_name        = 'biored',
              seed_value       = 1111,
              balance_ratio    = -1,
              use_single_chunk = False):
        
    # Define device: use GPU if available, else use CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps") # Multi-Process Service (MPS) is available
    else:    
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_processor = BioREDDataset if task_name == 'biored' else CDRDataset

    # Initialize pre-trained tokenizer
    tokenizer = load_tokenizer(in_bert_model, dataset_processor)

    # Initialize datasets
    train_dataset = dataset_processor(in_train_tsv_file,
                                      tokenizer,
                                      max_seq_len      = max_seq_len,
                                      soft_prompt_len  = soft_prompt_len,
                                      balance_ratio    = balance_ratio,
                                      use_single_chunk = use_single_chunk)
    # Create DataLoaders
    none_label_index = train_dataset.get_label_2_id('relation')['None']
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size  = batch_size, 
                                  collate_fn  = custom_collate_fn, 
                                  shuffle     = True,
                                  num_workers = 10, 
                                  pin_memory  = (device.type == 'cuda'))

    if in_dev_tsv_file != '':
        dev_dataset    = dataset_processor(in_dev_tsv_file,
                                           tokenizer,
                                           max_seq_len      = max_seq_len,
                                           soft_prompt_len  = soft_prompt_len,
                                           use_single_chunk = use_single_chunk)
        
        dev_dataloader = DataLoader(dev_dataset,   
                                    batch_size  = batch_size, 
                                    collate_fn  = custom_collate_fn,
                                    num_workers = 10, 
                                    pin_memory  = (device.type == 'cuda'))
    else:
        dev_dataloader = None
        
    # Instantiate the modified modelbert_model,     
    bioredirect_model = BioREDirect(in_bert_model   = in_bert_model, 
                                    soft_prompt_len = soft_prompt_len,
                                    relation_label_to_id  = train_dataset.get_label_2_id('relation'),
                                    novelty_label_to_id   = train_dataset.get_label_2_id('novelty'),
                                    direction_label_to_id = train_dataset.get_label_2_id('direction'),
                                    num_soft_prompt       = 20,
                                    use_single_chunk      = use_single_chunk,)
    bioredirect_model.resize_token_embeddings(len(tokenizer))
    bioredirect_model.to(device)
    optimizer = torch.optim.AdamW(bioredirect_model.parameters(), lr=learning_rate)

    # Create a BCEWithLogitsLoss object using the class weights
    loss_relation_func  = nn.BCEWithLogitsLoss(reduction='none')
    loss_novelty_func   = nn.BCEWithLogitsLoss(reduction='none')
    loss_direction_func = nn.BCEWithLogitsLoss(reduction='none')
    
    # Example training loop
    # Assuming the modified_model_with_classifier and other necessary components are already defined
    
    best_f1_score = None

    logger.info(f"Start training for {num_epochs} epochs")
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}")
        bioredirect_model.train()
        total_loss = 0
        total_batches = len(train_dataloader)
        for i, batch in tqdm(enumerate(train_dataloader), total=total_batches, desc="Training"):
            #if i == 20:
            #    break
            optimizer.zero_grad()
            #labels = batch['labels'].to(device)
            #print('===============>', batch.keys())
            gold_rel_labels = batch['relation_labels'].to(device)
            gold_nov_labels = batch['novelty_labels'].to(device)
            gold_dir_labels = batch['direction_labels'].to(device)
            (pred_rel_logits, 
             pred_nov_logits, 
             pred_dir_logits) = bioredirect_model(input_ids             = batch['input_ids'].to(device), 
                                                  attention_mask        = batch['attention_mask'].to(device),
                                                  #entity1_indices       = batch['entity1_indices'],
                                                  #entity2_indices       = batch['entity2_indices'],
                                                  #entity1_sent_ids      = batch['entity1_sent_ids'],
                                                  #entity2_sent_ids      = batch['entity2_sent_ids'],
                                                  pair_prompt_ids       = batch['pair_prompt_ids'].to(device),
                                                  #sent_ids              = batch['sent_ids'],
                                                  relation_token_index  = batch['relation_token_index'],
                                                  direction_token_index = batch['direction_token_index'],
                                                  novelty_token_index   = batch['novelty_token_index'])
            
            weighted_loss = 0
            if task_name == 'biored':
                rel_loss = loss_relation_func(pred_rel_logits, gold_rel_labels)

                weighted_loss = 0

                for j in range(batch_size):
                    if j < len(gold_rel_labels):
                        if gold_rel_labels[j][none_label_index] == 0:
                            # Use weighted loss for non-'None' relation
                            '''loss = (weight_rel_loss * rel_loss[j].mean() + 
                                    weight_nov_loss * nov_loss[j].mean() + 
                                    weight_dir_loss * dir_loss[j].mean())'''
                            loss = rel_loss[j].sum() 
                            if loss > 0:                                
                                nov_loss = loss_novelty_func(pred_nov_logits, gold_nov_labels)
                                dir_loss = loss_direction_func(pred_dir_logits, gold_dir_labels)
                                loss += (nov_loss[j].sum() / (rel_loss[j].sum() + nov_loss[j].sum() + dir_loss[j].sum())) * nov_loss[j].sum()
                                loss += (dir_loss[j].sum() / (rel_loss[j].sum() + nov_loss[j].sum() + dir_loss[j].sum())) * dir_loss[j].sum()
                        else:
                            # Use only rel_loss for 'None' relation
                            loss = rel_loss[j].sum()
                        
                        # Accumulate the loss for the batch
                        weighted_loss += loss

                # Average the loss over the batch
                weighted_loss = weighted_loss / batch_size
            else:
                weighted_loss = loss_relation_func(pred_rel_logits, gold_rel_labels).mean()
                weighted_loss = weighted_loss / batch_size
        
            weighted_loss.backward()
            all_loss = weighted_loss
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += all_loss.item()
        
        avg_train_loss = total_loss / total_batches
        logger.info(f"Epoch {epoch+1}, Train Loss: {avg_train_loss}")
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss}")
        
        
        # Evaluate F1-score on the validation set
        if dev_dataloader:
            logger.info("Evaluating F1-score on the validation set")
            if task_name == 'biored':
                val_out_dict = evaluate_biored_f1_score(bioredirect_model, 
                                                        dev_dataloader, 
                                                        none_label_index     = none_label_index, 
                                                        device               = device,
                                                        relation_label_list  = train_dataset.get_labels('relation'),
                                                        novelty_label_list   = train_dataset.get_labels('novelty'),
                                                        direction_label_list = train_dataset.get_labels('direction'))    
                # print out the results
                for category in ['TP', 'FP', 'FN', 'Precision', 'Recall', 'F1']: c = category.lower(); logger.info(f"Epoch {epoch + 1}, Validation {category}: {val_out_dict['pair'][c]} / {val_out_dict['rel'][c]} / {val_out_dict['pair_nov'][c]} / {val_out_dict['rel_nov'][c]} / {val_out_dict['rel_dir'][c]} / {val_out_dict['all'][c]} / {val_out_dict['relaxed_rel'][c]} / {val_out_dict['relaxed_rel_nov'][c]} / {val_out_dict['relaxed_rel_dir'][c]} / {val_out_dict['relaxed_all'][c]}")
                for category in ['TP', 'FP', 'FN', 'Precision', 'Recall', 'F1']: c = category.lower(); print(f"Epoch {epoch + 1}, Validation {category}: {val_out_dict['pair'][c]} / {val_out_dict['rel'][c]} / {val_out_dict['pair_nov'][c]} / {val_out_dict['rel_nov'][c]} / {val_out_dict['rel_dir'][c]} / {val_out_dict['all'][c]} / {val_out_dict['relaxed_rel'][c]} / {val_out_dict['relaxed_rel_nov'][c]} / {val_out_dict['relaxed_rel_dir'][c]} / {val_out_dict['relaxed_all'][c]}")
            
            else:
                val_out_dict = evaluate_cdr_f1_score(bioredirect_model, 
                                                     dev_dataloader, 
                                                     none_label_index     = none_label_index, 
                                                     device               = device,
                                                     label_list           = train_dataset.get_labels('relation'))
                # print out the results
                for category in ['TP', 'FP', 'FN', 'Precision', 'Recall', 'F1']: c = category.lower(); logger.info(f"Epoch {epoch + 1}, Validation {category}: {val_out_dict['all'][c]}")
                for category in ['TP', 'FP', 'FN', 'Precision', 'Recall', 'F1']: c = category.lower(); print(f"Epoch {epoch + 1}, Validation {category}: {val_out_dict['all'][c]}")
                
            # Save the model if it has the best F1-score so far
            if best_f1_score == None or val_out_dict['all']['f1'] > best_f1_score['f1']:
                best_f1_score = val_out_dict['all']
                
                bioredirect_model.save_model(out_bioredirect_model + '//out_bioredirect_model.pth')
                tokenizer.save_pretrained(out_bioredirect_model)

                logger.info(f"New best model saved with F1-score: {best_f1_score['f1']}")            
                print(f"New best model saved with F1-score: {best_f1_score['f1']}")
                with open(f'val_pred_{seed_value}_{soft_prompt_len}_{num_epochs}_{batch_size}_{max_seq_len}_{neg_loss_weight}_{learning_rate}_{balance_ratio}.tsv', 'w') as f: [f.write(line + '\n') for line in val_out_dict['out_tsv_str_list']]
        else:
            bioredirect_model.save_model(out_bioredirect_model + '//out_bioredirect_model.pth')
            tokenizer.save_pretrained(out_bioredirect_model)
            logger.info("Model saved")
            print("Model saved")

    logger.info("Training complete!")
    if dev_dataloader:
        logger.info(f"Best F1-Score: {best_f1_score['f1']}")

def run_inference(in_bioredirect_model,
                  in_test_tsv_file,
                  out_pred_tsv_file = '',
                  soft_prompt_len = 5,
                  num_epochs  = 10,
                  batch_size  = 16,
                  max_seq_len = 512,
                  task_name   = 'biored',
                  no_eval     = False,
                  use_single_chunk = False):
    
    # Define device: use GPU if available, else use CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps") # Multi-Process Service (MPS) is available
    else:    
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained(in_bioredirect_model)    
    bioredirect_model = BioREDirect.load_model(model_path = in_bioredirect_model)
    dataset_processor = BioREDDataset if task_name == 'biored' else CDRDataset
    test_dataset      = dataset_processor(in_test_tsv_file,
                                          tokenizer,
                                          max_seq_len      = max_seq_len,
                                          soft_prompt_len  = bioredirect_model.soft_prompt_len,
                                          use_single_chunk = use_single_chunk)
            
    none_label_index  = test_dataset.get_label_2_id('relation')['None'] # 'None' label index is the same for all tasks

    test_dataloader   = DataLoader(test_dataset, 
                                   batch_size  = batch_size, 
                                   collate_fn  = custom_collate_fn,
                                   num_workers = 4, 
                                   pin_memory = (device.type == 'cuda'))
    if task_name == 'biored':
        test_out_dict = evaluate_biored_f1_score(bioredirect_model, 
                                                 test_dataloader, 
                                                 none_label_index     = none_label_index, 
                                                 device               = device,
                                                 relation_label_list  = test_dataset.get_labels('relation'),
                                                 novelty_label_list   = test_dataset.get_labels('novelty'),
                                                 direction_label_list = test_dataset.get_labels('direction'),
                                                 no_eval              = no_eval)
    else:
        test_out_dict = evaluate_cdr_f1_score(bioredirect_model, 
                                              test_dataloader, 
                                              none_label_index     = none_label_index, 
                                              device               = device,
                                              label_list           = test_dataset.get_labels('relation'),
                                              no_eval              = no_eval)
    
    if not no_eval:
        if task_name == 'biored':
            logger.info(f"Test: P / R / P+N / R+N / R+D / All / Relaxed R / Relaxed R+N / Relaxed R+D / Relaxed All")
            print(f"Test: P / R / P+N / R+N / R+D / All / Relaxed R / Relaxed R+N / Relaxed R+D / Relaxed All")
            for category in ['TP', 'FP', 'FN', 'Precision', 'Recall', 'F1']: c = category.lower(); logger.info(f"Test {category}: {test_out_dict['pair'][c]} / {test_out_dict['rel'][c]} / {test_out_dict['pair_nov'][c]} / {test_out_dict['rel_nov'][c]} / {test_out_dict['rel_dir'][c]} / {test_out_dict['all'][c]} / {test_out_dict['relaxed_rel'][c]} / {test_out_dict['relaxed_rel_nov'][c]} / {test_out_dict['relaxed_rel_dir'][c]} / {test_out_dict['relaxed_all'][c]}")
            for category in ['TP', 'FP', 'FN', 'Precision', 'Recall', 'F1']: c = category.lower(); print(f"Test {category}: {test_out_dict['pair'][c]} / {test_out_dict['rel'][c]} / {test_out_dict['pair_nov'][c]} / {test_out_dict['rel_nov'][c]} / {test_out_dict['rel_dir'][c]} / {test_out_dict['all'][c]} / {test_out_dict['relaxed_rel'][c]} / {test_out_dict['relaxed_rel_nov'][c]} / {test_out_dict['relaxed_rel_dir'][c]} / {test_out_dict['relaxed_all'][c]}")
        else:
            for category in ['TP', 'FP', 'FN', 'Precision', 'Recall', 'F1']: c = category.lower(); logger.info(f"Test {category}: {test_out_dict['all'][c]}")
            for category in ['TP', 'FP', 'FN', 'Precision', 'Recall', 'F1']: c = category.lower(); print(f"Test {category}: {test_out_dict['all'][c]}")

    if out_pred_tsv_file != '':
        with open(out_pred_tsv_file, 'w') as f: [f.write(line + '\n') for line in test_out_dict['out_tsv_str_list']]
        with open(out_pred_tsv_file + '.pred.label.tsv', 'w') as f: [f.write('\t'.join(pred) + '\n') for pred in test_out_dict['pred_labels']]
    else:
        with open(f'test_pred_{seed_value}_{soft_prompt_len}_{num_epochs}_{batch_size}_{max_seq_len}_{neg_loss_weight}_{learning_rate}_{balance_ratio}.tsv', 'w') as f: [f.write(line + '\n') for line in test_out_dict['out_tsv_str_list']]
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Run Relation Extraction Experiment')
    parser.add_argument('--seed', type=int, default=1111, help='Seed value')
    parser.add_argument('--in_bert_model', type=str, default='biorex_biolinkbert_pt', help='Input BERT model name')
    parser.add_argument('--out_bioredirect_model', type=str, default='', help='Output BERT model name')
    parser.add_argument('--in_bioredirect_model', type=str, default='', help='Output BERT model name')
    parser.add_argument('--in_train_tsv_file', type=str, default='', help='Input train dataset path')
    parser.add_argument('--in_dev_tsv_file', type=str, default='', help='Input dev dataset path')
    parser.add_argument('--in_test_tsv_file', type=str, default='', help='Input test dataset path')
    parser.add_argument('--out_pred_tsv_file', type=str, default='', help='Output prediction tsv path')
    parser.add_argument('--soft_prompt_len', type=int, default=32, help='Soft Prompt length')
    parser.add_argument('--num_labels', type=int, default=9, help='Number of labels')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--max_seq_len', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--neg_loss_weight', type=float, default=0.2, help='Negative loss weight')
    parser.add_argument('--learning_rate', type=float, default=5e-6, help='Learning rate')
    parser.add_argument('--task_name', type=str, default='biored', help='Task name')
    parser.add_argument('--is_multi_label', type=bool, default=False, help='Is multi-label')
    parser.add_argument('--balance_ratio', type=float, default=-1, help='Balance ratio')
    parser.add_argument('--no_eval', type=bool, default=False, help='No evaluation')
    parser.add_argument('--use_single_chunk', type=bool, default=False, help='Use single chunk for training')

    args = parser.parse_args() 

    # Set the seed value
    seed_value = args.seed
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # If using multiple GPUs
    set_seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_default_dtype(torch.float32)

    in_bert_model         = args.in_bert_model
    out_bioredirect_model = args.out_bioredirect_model

    soft_prompt_len       = args.soft_prompt_len
    num_epochs            = args.num_epochs
    batch_size            = args.batch_size
    max_seq_len           = args.max_seq_len
    neg_loss_weight       = args.neg_loss_weight
    learning_rate         = args.learning_rate
    task_name             = args.task_name
    is_multi_label        = args.is_multi_label
    balance_ratio         = args.balance_ratio
    no_eval               = args.no_eval
    use_single_chunk      = args.use_single_chunk

    print('================>args.no_eval', args.no_eval)

    if args.in_train_tsv_file != '':
        run_train(in_bert_model         = in_bert_model, 
                  out_bioredirect_model = out_bioredirect_model, 
                  in_train_tsv_file     = args.in_train_tsv_file,
                  in_dev_tsv_file       = args.in_dev_tsv_file,
                  soft_prompt_len       = soft_prompt_len,
                  num_epochs            = num_epochs,
                  batch_size            = batch_size,
                  max_seq_len           = max_seq_len,
                  neg_loss_weight       = neg_loss_weight,
                  learning_rate         = learning_rate,
                  task_name             = task_name,
                  seed_value            = seed_value,
                  balance_ratio         = balance_ratio,
                  use_single_chunk      = use_single_chunk)
    
    if args.in_test_tsv_file != '':
        run_inference(in_bioredirect_model  = args.in_bioredirect_model if args.in_bioredirect_model != '' else args.out_bioredirect_model,
                      in_test_tsv_file      = args.in_test_tsv_file,
                      out_pred_tsv_file     = args.out_pred_tsv_file,
                      soft_prompt_len  = soft_prompt_len,
                      num_epochs       = num_epochs,
                      batch_size       = batch_size,
                      max_seq_len      = max_seq_len,
                      task_name        = task_name,
                      no_eval          = no_eval,
                      use_single_chunk = use_single_chunk)
