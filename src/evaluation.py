import numpy as np
import torch
from tqdm import tqdm
import copy
def precision_recall_f1(y_true, y_pred, class_label):
    true_positives = np.sum((y_pred == class_label) & (y_true == class_label))
    predicted_positives = np.sum(y_pred == class_label)
    actual_positives = np.sum(y_true == class_label)
    
    precision = true_positives / predicted_positives if predicted_positives > 0 else 0
    recall = true_positives / actual_positives if actual_positives > 0 else 0
    
    return precision, recall

def f1_score_with_details(y_true, y_pred, labels):
    tp = 0.
    fp = 0.
    fn = 0.
    
    for gold, pred in zip(y_true, y_pred):
        if gold == pred and gold in labels:
            tp += 1
        elif gold != pred:
            if gold in labels:
                fn += 1
            if pred in labels:
                fp += 1
    
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    scores = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }
    return scores

def relaxed_f1_score_with_details(y_true, y_pred, labels, pos_label, neg_label, ass_label):
    tp = 0.
    fp = 0.
    fn = 0.
    
    for gold, pred in zip(y_true, y_pred):
        if ((gold == pos_label) or (gold == neg_label)) and pred == ass_label:
            tp += 1
        elif gold == pred and gold in labels:
            tp += 1
        elif gold != pred:
            if gold in labels:
                fn += 1
            if pred in labels:
                fp += 1
    
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    scores = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }
    return scores

def f1_score_novelty_with_details(y_true, y_pred, labels, none_label):
    tp = 0.
    fp = 0.
    fn = 0.
    
    for gold, pred in zip(y_true, y_pred):

        if gold[0] == pred[0] and gold[1] == pred[1] and gold[0] != none_label:
            tp += 1
            #print('tp', gold, pred)
        elif gold[0] != pred[0] or gold[1] != pred[1]:
            if gold[0] in labels:
                fn += 1
                #print('fn', gold, pred)
            if pred[0] in labels:
                fp += 1
                #print('fp', gold, pred)
    
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    scores = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }
    return scores

def relaxed_f1_score_novelty_with_details(y_true, y_pred, labels, none_label, pos_label, neg_label, ass_label):
    tp = 0.
    fp = 0.
    fn = 0.
    
    # gold[0]: rel_label
    # gold[1]: novelty_label
    for gold, pred in zip(y_true, y_pred):

        if (gold[0] == pos_label or gold[0] == neg_label) and pred[0] == ass_label:
            if gold[1] == pred[1]:
                tp += 1
            else:
                fn += 1
                fp += 1
        elif gold[0] == pred[0] and gold[1] == pred[1] and gold[0] != none_label:
            tp += 1
        elif gold[0] != pred[0] or gold[1] != pred[1]:
            if gold[0] in labels:
                fn += 1
            if pred[0] in labels:
                fp += 1
    
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    scores = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }
    return scores

def merge_labels(labels, labels_to_merge, new_label):
    merged_labels = np.copy(labels)
    '''for l, ml in zip(labels, merged_labels):
        print(l, ml)
    print(labels_to_merge)
    print(new_label)'''
    for label_to_merge in labels_to_merge:
        merged_labels[labels == label_to_merge] = new_label
    return merged_labels

def merge_labels_with_novelty(labels, labels_to_merge, new_label):
    merged_labels = copy.deepcopy(labels)
    '''for l, ml in zip(labels, merged_labels):
        print(l, ml)'''
        
    for i, (rel, nov) in enumerate(merged_labels):

        if rel in labels_to_merge:
            merged_labels[i] = (new_label, nov)
            #print('merged_labels[i]', merged_labels[i], new_label)
    return merged_labels

def convert_to_single_label(label_vectors, 
                            label_list):
    single_labels = []
    
    # Define index for 'None' and 'Association' based on their positions in label_list
    none_index = label_list.index('None')
    association_index = label_list.index('Association')
    novel_index = label_list.index('Novel')
    no_novel_index = label_list.index('No')
    
    for vector in label_vectors:
        if sum(vector) == 0:
            # If all entries are zero, label is 'None'
            #single_labels.append('None')
            single_labels.append(none_index)
        else:
            # Check for any label other than 'None' or 'Association'
            high_priority_label = None
            max_score = 0.5
            #print(vector)
            for i, label_present in enumerate(vector):
                if label_present > max_score and i != none_index and i != association_index and i != novel_index and i != no_novel_index:
                    #high_priority_label = label_list[i]
                    high_priority_label = i
                    max_score = label_present
            #print('high_priority_label', high_priority_label)
            #print('max_score', max_score)
            
            if high_priority_label:
                # If a label other than 'None' or 'Association' is found
                single_labels.append(high_priority_label)
            elif vector[association_index] > 0.5:
                # If 'Association' is the only label above the threshold
                single_labels.append(association_index)
            elif vector[none_index] > 0.5:
                # If 'None' is the only label above the threshold
                single_labels.append(none_index)
            else:
                # If no label is above the threshold, pick the label with the highest score
                max_index = none_index
                max_score = vector[none_index]
                #print(vector)
                for i, label_present in enumerate(vector):
                    if label_present > max_score and i != association_index and i != novel_index and i != no_novel_index:
                        #high_priority_label = label_list[i]
                        max_index = i
                        max_score = label_present
                single_labels.append(max_index)
    
    return single_labels

def convert_to_biored_label(relation_label_vectors, 
                            novelty_label_vectors,
                            direction_label_vectors,
                            relation_label_list,
                            novelty_label_list,
                            direction_label_list):
    single_labels = []
    
    # Define index for 'None' and 'Association' based on their positions in label_list
    none_index         = relation_label_list.index('None') # 'None' is the same index in relation_label_list, novelty_label_list, and direction_label_list
    association_index  = relation_label_list.index('Association')
    pos_index          = relation_label_list.index('Positive_Correlation')
    neg_index          = relation_label_list.index('Negative_Correlation')
    conversion_index   = relation_label_list.index('Conversion')
    novel_index        = novelty_label_list.index('Novel')
    no_novel_index     = novelty_label_list.index('No')
    left_2_right_index = direction_label_list.index('Left_to_Right')
    right_2_left_index = direction_label_list.index('Right_to_Left')
    no_direct_index    = direction_label_list.index('No_Direct')
    subject_label      = no_direct_index
    
    for relation_label_vector, novelty_label_vector, direction_label_vector in zip(relation_label_vectors, 
                                                                                   novelty_label_vectors, 
                                                                                   direction_label_vectors):
        if sum(relation_label_vector) == 0:
            # If all entries are zero, label is 'None'
            #single_labels.append('None')
            single_labels.append((none_index, novel_index, subject_label))
        else:
            _novel_label = novel_index
            if novelty_label_vector[novel_index] < novelty_label_vector[no_novel_index]:
                _novel_label = no_novel_index
            
            _max_score = 0.0
            subject_label = left_2_right_index
            if direction_label_vector[left_2_right_index] >= _max_score:
                _max_score = direction_label_vector[left_2_right_index]
                subject_label = left_2_right_index
            if direction_label_vector[right_2_left_index] >= _max_score:
                _max_score = direction_label_vector[right_2_left_index]
                subject_label = right_2_left_index   
            if direction_label_vector[no_direct_index] >= _max_score:
                _max_score = direction_label_vector[no_direct_index]
                subject_label = no_direct_index
            
            # Check for any label other than 'None' or 'Association'
            high_priority_label = None

            max_score = 0.5
            for i, label_present in enumerate(relation_label_vector):
                if (label_present > max_score and 
                    i != none_index and 
                    i != association_index):                    
                    high_priority_label = i
                    max_score = label_present
            
            if high_priority_label:
                single_labels.append((high_priority_label, _novel_label, subject_label))
            else:
                # If no label is above the threshold, pick the label with the highest score
                max_index = none_index
                max_score = relation_label_vector[none_index]
                #print(vector)
                for i, label_present in enumerate(relation_label_vector):
                    if label_present > max_score:
                        max_index = i
                        max_score = label_present
                single_labels.append((max_index, _novel_label, subject_label))
    
    return single_labels

def convert_to_biored_label_with_score(relation_label_vectors, 
                                       novelty_label_vectors,
                                       direction_label_vectors,
                                       relation_label_list,
                                       novelty_label_list,
                                       direction_label_list):
    single_labels = []

    # Define index for 'None' and 'Association' based on their positions in label_list
    none_index         = relation_label_list.index('None') # 'None' is the same index in relation_label_list, novelty_label_list, and direction_label_list
    association_index  = relation_label_list.index('Association')
    pos_index          = relation_label_list.index('Positive_Correlation')
    neg_index          = relation_label_list.index('Negative_Correlation')
    conversion_index   = relation_label_list.index('Conversion')
    novel_index        = novelty_label_list.index('Novel')
    no_novel_index     = novelty_label_list.index('No')
    single_labels = []
    
    # Define index for 'None' and 'Association' based on their positions in label_list
    none_index         = relation_label_list.index('None') # 'None' is the same index in relation_label_list, novelty_label_list, and direction_label_list
    association_index  = relation_label_list.index('Association')
    pos_index          = relation_label_list.index('Positive_Correlation')
    neg_index          = relation_label_list.index('Negative_Correlation')
    conversion_index   = relation_label_list.index('Conversion')
    novel_index        = novelty_label_list.index('Novel')
    no_novel_index     = novelty_label_list.index('No')
    left_2_right_index = direction_label_list.index('Left_to_Right')
    right_2_left_index = direction_label_list.index('Right_to_Left')
    no_direct_index    = direction_label_list.index('No_Direct')
    subject_label      = no_direct_index
    
    for relation_label_vector, novelty_label_vector, direction_label_vector in zip(relation_label_vectors, 
                                                                                   novelty_label_vectors, 
                                                                                   direction_label_vectors):
        _rel_score, _nov_score, _dir_score = 0.0, 0.0, 0.0
        if sum(relation_label_vector) == 0:
            # If all entries are zero, label is 'None'
            #single_labels.append('None')
            single_labels.append((none_index, novel_index, subject_label, _rel_score, _nov_score, _dir_score))
        else:
            _novel_label = novel_index
            _nov_score = novelty_label_vector[novel_index]
            if novelty_label_vector[novel_index] < novelty_label_vector[no_novel_index]:
                _novel_label = no_novel_index
                _nov_score = novelty_label_vector[no_novel_index]
            
            _max_score = 0.0
            subject_label = left_2_right_index
            if direction_label_vector[left_2_right_index] >= _max_score:
                _max_score = direction_label_vector[left_2_right_index]
                subject_label = left_2_right_index
            if direction_label_vector[right_2_left_index] >= _max_score:
                _max_score = direction_label_vector[right_2_left_index]
                subject_label = right_2_left_index   
            if direction_label_vector[no_direct_index] >= _max_score:
                _max_score = direction_label_vector[no_direct_index]
                subject_label = no_direct_index
            _dir_score = _max_score
            
            # Check for any label other than 'None' or 'Association'
            high_priority_label = None

            max_score = 0.5
            for i, label_present in enumerate(relation_label_vector):
                if (label_present > max_score and 
                    i != none_index and 
                    i != association_index):                    
                    high_priority_label = i
                    max_score = label_present
            _rel_score = max_score
            
            if high_priority_label:
                single_labels.append((high_priority_label, _novel_label, subject_label, _rel_score, _nov_score, _dir_score))
            else:
                # If no label is above the threshold, pick the label with the highest score
                max_index = none_index
                max_score = relation_label_vector[none_index]
                #print(vector)
                for i, label_present in enumerate(relation_label_vector):
                    if label_present > max_score:
                        max_index = i
                        max_score = label_present
                _rel_score = max_score
                single_labels.append((max_index, _novel_label, subject_label, _rel_score, _nov_score, _dir_score))
    
    return single_labels

def convert_to_cdr_label(relation_label_vectors):
    single_labels = []
    
    for relation_label_vector in relation_label_vectors:
        max_score = 0.0
        label = 0

        for i, label_present in enumerate(relation_label_vector):
            if label_present > max_score:                    
                label = i
                max_score = label_present
        
        single_labels.append(label)
    
    return single_labels

def evaluate_biored_f1_score(model, 
                             dataloader, 
                             none_label_index, 
                             device,
                             relation_label_list=[],
                             novelty_label_list=[],
                             direction_label_list=[],
                             no_eval=False):
    
    # Convert the "None" label string to its integer representation
    
    out_tsv_str_list = []
    out_header = '\t'.join(relation_label_list + novelty_label_list + direction_label_list).strip('\t')
    out_tsv_str_list.append(out_header)
    gold_labels = []
    pred_labels = []
    model.eval()
    pos_label_index = relation_label_list.index('Positive_Correlation')
    neg_label_index = relation_label_list.index('Negative_Correlation')
    ass_label_index = relation_label_list.index('Association')
    with torch.no_grad():
        total_batches = len(dataloader)
        for i, batch in tqdm(enumerate(dataloader), total=total_batches, desc="Evaluation"):
            
            #if i == 20:
            #    break
            gold_rel_labels = batch['relation_labels'].numpy() if batch['relation_labels'].device.type == 'cpu' else batch['relation_labels'].cpu().numpy()
            gold_nov_labels = batch['novelty_labels'].numpy() if batch['novelty_labels'].device.type == 'cpu' else batch['novelty_labels'].cpu().numpy()
            gold_dir_labels = batch['direction_labels'].numpy() if batch['direction_labels'].device.type == 'cpu' else batch['direction_labels'].cpu().numpy()
            (rel_token_outputs, 
             nov_token_outputs, 
             dir_token_outputs) = model(input_ids             = batch['input_ids'].to(device), 
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

            pred_rel_scores = torch.sigmoid(rel_token_outputs).cpu().numpy()
            pred_nov_scores = torch.sigmoid(nov_token_outputs).cpu().numpy()
            pred_dir_scores = torch.sigmoid(dir_token_outputs).cpu().numpy()

            for rel, nov, dire in zip(pred_rel_scores, pred_nov_scores, pred_dir_scores):
                out_pred_str = '\t'.join(str(_score) for _score in np.concatenate((rel, nov, dire))).strip('\t')
                out_tsv_str_list.append(out_pred_str)

            #cls_preds = np.array(convert_to_biored_label(cls_scores, label_list))
            #pair_preds = np.array(convert_to_biored_label(pair_scores, label_list))
            _pred_labels = np.array(convert_to_biored_label(pred_rel_scores, 
                                                            pred_nov_scores,
                                                            pred_dir_scores,
                                                            relation_label_list,
                                                            novelty_label_list,
                                                            direction_label_list))
            
            _gold_labels = np.array(convert_to_biored_label(gold_rel_labels, 
                                                            gold_nov_labels,
                                                            gold_dir_labels,
                                                            relation_label_list,
                                                            novelty_label_list,
                                                            direction_label_list))
            
            gold_labels.extend(_gold_labels)
            pred_labels.extend(_pred_labels)
    
    out_pred_labels = [[relation_label_list[rel_label], novelty_label_list[nov_label], direction_label_list[subj_label]] for rel_label, nov_label, subj_label in pred_labels]

    if not no_eval:
        # Prepare labels for F1 calculation by excluding "None"
        # First, filter out instances where true label is "None"
        unique_labels = list(set([l for l, nl, sl in gold_labels]))
        if none_label_index in unique_labels:
            unique_labels.remove(none_label_index)
            
        rel_gold_labels = [l for l, nl, sl in gold_labels]
        rel_pred_labels = [l for l, nl, sl in pred_labels]

        rel_f1 = f1_score_with_details(rel_gold_labels, rel_pred_labels, labels=unique_labels)
        rel_f1_relaxed = relaxed_f1_score_with_details(rel_gold_labels, 
                                                    rel_pred_labels, 
                                                    unique_labels,
                                                    pos_label_index,
                                                    neg_label_index,
                                                    ass_label_index)

        rel_nov_gold_labels = [(l, nl) for l, nl, sl in gold_labels]
        rel_nov_pred_labels = [(l, nl) for l, nl, sl in pred_labels]
        rel_nov_f1 = f1_score_novelty_with_details(rel_nov_gold_labels, rel_nov_pred_labels, unique_labels, none_label_index)
        rel_nov_f1_relaxed = relaxed_f1_score_novelty_with_details(gold_labels, 
                                                                pred_labels, 
                                                                unique_labels,
                                                                none_label_index,
                                                                pos_label_index,
                                                                neg_label_index,
                                                                ass_label_index)
        
        rel_subj_gold_labels = [(l, sl) for l, nl, sl in gold_labels]
        rel_subj_pred_labels = [(l, sl) for l, nl, sl in pred_labels]
        rel_subj_f1 = f1_score_novelty_with_details(rel_subj_gold_labels, rel_subj_pred_labels, unique_labels, none_label_index)
        rel_subj_f1_relaxed = relaxed_f1_score_novelty_with_details(gold_labels, 
                                                                    pred_labels, 
                                                                    unique_labels,
                                                                    none_label_index,
                                                                    pos_label_index,
                                                                    neg_label_index,
                                                                    ass_label_index)
        
        all_gold_labels = [(l, str(nl) + '_' + str(sl)) for l, nl, sl in gold_labels]
        all_pred_labels = [(l, str(nl) + '_' + str(sl)) for l, nl, sl in pred_labels]
        
        all_f1 = f1_score_novelty_with_details(all_gold_labels, all_pred_labels, unique_labels, none_label_index)
        all_f1_relaxed = relaxed_f1_score_novelty_with_details(gold_labels, 
                                                            pred_labels, 
                                                            unique_labels,
                                                            none_label_index,
                                                            pos_label_index,
                                                            neg_label_index,
                                                            ass_label_index)
        
        new_label = relation_label_list.index('Association')  # New label for positive class
        merged_gold_labels = merge_labels(rel_gold_labels, unique_labels, new_label)
        merged_predicted_labels = merge_labels(rel_pred_labels, unique_labels, new_label)
        pair_f1 = f1_score_with_details(merged_gold_labels, merged_predicted_labels, labels=[new_label])

        gold_labels = [(l, nl) for l, nl, sl in gold_labels]
        pred_labels = [(l, nl) for l, nl, sl in pred_labels]
        merged_gold_labels = merge_labels_with_novelty(gold_labels, unique_labels, new_label)
        merged_pred_labels = merge_labels_with_novelty(pred_labels, unique_labels, new_label)
        pair_novelty_f1 = f1_score_novelty_with_details(merged_gold_labels, merged_pred_labels, unique_labels, none_label_index)
        
        out_dict = {
            'pair': pair_f1,
            'pair_nov': pair_novelty_f1,
            'rel': rel_f1,
            'rel_nov': rel_nov_f1,
            'rel_dir': rel_subj_f1,
            'all': all_f1,
            'relaxed_rel': rel_f1_relaxed,
            'relaxed_rel_nov': rel_nov_f1_relaxed,
            'relaxed_rel_dir': rel_subj_f1_relaxed,
            'relaxed_all': all_f1_relaxed,
            'out_tsv_str_list': out_tsv_str_list,
            'pred_labels': out_pred_labels
        }
    else:
        out_dict = {
            'out_tsv_str_list': out_tsv_str_list,
            'pred_labels': out_pred_labels
        }
    return out_dict

def evaluate_cdr_f1_score(model, 
                          dataloader, 
                          none_label_index, 
                          device,
                          label_list=[],
                          no_eval=False):
    
    # Convert the "None" label string to its integer representation
    
    out_tsv_str_list = []
    out_header = '\t'.join(label_list).strip('\t')
    out_tsv_str_list.append(out_header)
    gold_labels = []
    pred_labels = []
    model.eval()
    with torch.no_grad():
        total_batches = len(dataloader)
        for i, batch in tqdm(enumerate(dataloader), total=total_batches, desc="Evaluation"):
            
            #if i == 20:
            #    break
            gold_rel_labels = batch['relation_labels'].cpu().numpy()  # True labels
            (rel_token_outputs, _, _) = model(input_ids             = batch['input_ids'].to(device), 
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

            #pred_rel_scores = (torch.sigmoid(rel_token_outputs).cpu().numpy() + torch.sigmoid(pair_outputs).cpu().numpy())/2
            pred_rel_scores = torch.sigmoid(rel_token_outputs).cpu().numpy()
            #pred_rel_scores = torch.sigmoid(pair_outputs).cpu().numpy()
            

            for rel in zip(pred_rel_scores):
                out_pred_str = '\t'.join(str(_score) for _score in rel).strip('\t')
                out_tsv_str_list.append(out_pred_str)

            #cls_preds = np.array(convert_to_biored_label(cls_scores, label_list))
            #pair_preds = np.array(convert_to_biored_label(pair_scores, label_list))
            _pred_labels = np.array(convert_to_cdr_label(pred_rel_scores))
            _gold_labels = np.array(convert_to_cdr_label(gold_rel_labels))
            
            gold_labels.extend(_gold_labels)
            pred_labels.extend(_pred_labels)
    
    # Prepare labels for F1 calculation by excluding "None"
    # First, filter out instances where true label is "None"
    if not no_eval:
        unique_labels = list(set([l for l in gold_labels]))
        if none_label_index in unique_labels:
            unique_labels.remove(none_label_index)
            
        rel_gold_labels = [l for l in gold_labels]
        rel_pred_labels = [l for l in pred_labels]

        rel_f1 = f1_score_with_details(rel_gold_labels, rel_pred_labels, labels=unique_labels)

        out_dict = {
            'all': rel_f1,
            'out_tsv_str_list': out_tsv_str_list
        }
    else:
        out_dict = {
            'out_tsv_str_list': out_tsv_str_list
        }
    return out_dict

