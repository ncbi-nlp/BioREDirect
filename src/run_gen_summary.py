import glob

def load_pred_labels(in_pred_tsv_file,
                     level,
                     threshold = 0.0):
    pred_labels = []
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
    
    with open(in_pred_tsv_file, 'r') as f:
        f.readline()
        for line in f:
            tks = line.rstrip().split('\t')
            rel_scores = []
            for i in range(0, len(relation_labels)):
                rel_scores.append(float(tks[i]))
                
            nov_scores = []
            for i in range(len(relation_labels), len(relation_labels) + len(novelty_labels)):
                nov_scores.append(float(tks[i]))

            dir_scores = []
            for i in range(len(relation_labels) + len(novelty_labels), len(relation_labels) + len(novelty_labels) + len(direction_labels)):
                dir_scores.append(float(tks[i]))

            if threshold == 0:
                max_score = 0
                rel_label = ''

                for score, label in zip(rel_scores, relation_labels):
                    if score >= 0.5 and (label != 'Association' and label != 'None'):
                        rel_label = label
                        max_score = score
                if rel_label == '':
                    for score, label in zip(rel_scores, relation_labels):
                        if score >= max_score:
                            rel_label = label
                            max_score = score
            else:
                max_score = 0.5
                rel_label = ''
                for score, label in zip(rel_scores, relation_labels):
                    if label != 'None' and label != 'Association' and score >= max_score:
                        rel_label = label
                        max_score = score
                if rel_label == '':
                    max_score = 0
                    for score, label in zip(rel_scores, relation_labels):
                        if score >= max_score:
                            rel_label = label
                            max_score = score
            max_score = 0
            max_nov_label = ''
            max_pos_nov_score = 0
            max_pos_nov_label = ''
            for score, label in zip(nov_scores, novelty_labels):
                if score >= max_score:
                    nov_label = label
                    max_score = score
                if label != 'None' and score >= max_pos_nov_score:
                    max_pos_nov_score = score
                    max_pos_nov_label = label
            max_score = 0
            dir_label = ''
            max_pos_dir_score = 0
            max_pos_dir_label = ''
            for score, label in zip(dir_scores, direction_labels):
                if score >= max_score:
                    dir_label = label
                    max_score = score
                if label != 'None' and score >= max_pos_dir_score:
                    max_pos_dir_score = score
                    max_pos_dir_label = label
            
            if rel_label != 'None':
                if nov_label == 'None':
                    nov_label = max_pos_nov_label
                if dir_label == 'None':
                    dir_label = max_pos_dir_label

            if level == 'pair':
                rel_label = 'Association' if rel_label != 'None' else 'None'
                pred_labels.append((rel_label, '_'))
            elif level == 'relation':
                pred_labels.append((rel_label, '_'))
            elif level == 'pair+novelty':
                rel_label = 'Association' if rel_label != 'None' else 'None'
                pred_labels.append((rel_label, nov_label))
            elif level == 'relation+novelty':
                pred_labels.append((rel_label, nov_label))
            elif level == 'relation+direction':
                pred_labels.append((rel_label, dir_label))
            elif level == 'all':
                pred_labels.append((rel_label, nov_label, dir_label))

    return pred_labels

def load_gold_labels(in_gold_tsv_file,
                     level):

    gold_labels = []

    with open(in_gold_tsv_file, 'r') as f:
        for line in f:
            tks = line.rstrip().split('\t')
            rel_label = tks[-3]
            nov_label = tks[-2]
            sub_id    = tks[-1]
            id1       = tks[3]
            id2       = tks[4]
            dir_label = 'None'
            if rel_label != 'None':
                if sub_id == id1:
                    dir_label = 'Left_to_Right'
                elif sub_id == id2:
                    dir_label = 'Right_to_Left'
                else:
                    dir_label = 'No_Direct'
            
            if level == 'pair':
                rel_label = 'Association' if rel_label != 'None' else 'None'
                gold_labels.append((rel_label, '_'))
            elif level == 'relation':
                gold_labels.append((rel_label, '_'))
            elif level == 'pair+novelty':
                rel_label = 'Association' if rel_label != 'None' else 'None'
                gold_labels.append((rel_label, nov_label))
            elif level == 'relation+novelty':
                gold_labels.append((rel_label, nov_label))
            elif level == 'relation+direction':
                gold_labels.append((rel_label, dir_label))
            elif level == 'all':
                gold_labels.append((rel_label, nov_label, dir_label))

    return gold_labels

def eval(pred_labels, gold_labels):
    #print(len(pred_labels), len(gold_labels))
    assert len(pred_labels) == len(gold_labels)

    tp = 0.
    fp = 0.
    fn = 0.
    for pred, gold in zip(pred_labels, gold_labels):
        #print(pred, gold)
        if gold[0] != 'None':
            if pred == gold:
                tp += 1
            elif pred[0] == 'None':
                fn += 1
            else:
                fp += 1
                fn += 1
        elif pred[0] != 'None':
            fp += 1

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall    = tp / (tp + fn) if tp + fn > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return {'precision': precision, 'recall': recall, 'f1': f1}

def run_eval(in_gold_tsv_file,
             in_pred_tsv_file_pattern,
             out_tsv_file,
             threshold = 0.0):

    config_2_scores = {}
    count_pred_labels = None
    count_pred_direction = None
    count_pred_novelty = None
    for in_pred_tsv_file in glob.glob(in_pred_tsv_file_pattern):
        #print(in_pred_tsv_file)
        _, _, seed, config = in_pred_tsv_file.split('\\')[-1].split('_', 3)
        print(seed, config)
        config = config.replace('.tsv', '')
        if config not in config_2_scores:
            config_2_scores[config] = []
        
        #print(in_pred_tsv_file, in_gold_tsv_file)
        pred_labels = load_pred_labels(in_pred_tsv_file, level = 'pair', threshold = threshold)
        gold_labels = load_gold_labels(in_gold_tsv_file, level = 'pair')
        pair_f1 = eval(pred_labels, gold_labels)
        
        pred_labels = load_pred_labels(in_pred_tsv_file, level = 'pair+novelty', threshold = threshold)
        gold_labels = load_gold_labels(in_gold_tsv_file, level = 'pair+novelty')
        pair_novelty_f1 = eval(pred_labels, gold_labels)

        pred_labels = load_pred_labels(in_pred_tsv_file, level = 'relation', threshold = threshold)
        gold_labels = load_gold_labels(in_gold_tsv_file, level = 'relation')
        relation_f1 = eval(pred_labels, gold_labels)

        pred_labels = load_pred_labels(in_pred_tsv_file, level = 'relation+novelty', threshold = threshold)
        gold_labels = load_gold_labels(in_gold_tsv_file, level = 'relation+novelty')
        relation_novelty_f1 = eval(pred_labels, gold_labels)

        pred_labels = load_pred_labels(in_pred_tsv_file, level = 'relation+direction', threshold = threshold)
        gold_labels = load_gold_labels(in_gold_tsv_file, level = 'relation+direction')
        relation_direction_f1 = eval(pred_labels, gold_labels)

        pred_labels = load_pred_labels(in_pred_tsv_file, level = 'all', threshold = threshold)
        gold_labels = load_gold_labels(in_gold_tsv_file, level = 'all')
        all_direction_f1 = eval(pred_labels, gold_labels)

        if count_pred_labels is None:
            count_pred_labels = []
            count_pred_direction = []
            count_pred_novelty = []
            for i in range(len(pred_labels)):
                count_pred_labels.append({'None': 0, 'Association': 0, 'Bind': 0, 'Comparison': 0, 'Conversion': 0, 'Cotreatment': 0, 'Drug_Interaction': 0, 'Negative_Correlation': 0, 'Positive_Correlation': 0})
                count_pred_direction.append({'None': 0, 'Left_to_Right': 0, 'Right_to_Left': 0, 'No_Direct': 0})
                count_pred_novelty.append({'None': 0, 'Novel': 0, 'No': 0})
        for i, (pred_rel, pred_nov, pred_dir) in enumerate(pred_labels):
            count_pred_labels[i][pred_rel] += 1
            count_pred_direction[i][pred_dir] += 1
            count_pred_novelty[i][pred_nov] += 1                

        config_2_scores[config].append((pair_f1, pair_novelty_f1, relation_f1, relation_novelty_f1, relation_direction_f1, all_direction_f1))
        #print(pair_f1, pair_novelty_f1, relation_f1, relation_novelty_f1, relation_direction_f1, all_direction_f1)
    
    ensemble_pred_labels = []
    num_config = len(config_2_scores)
    for rel_votes, dir_votes, nov_votes in zip(count_pred_labels, count_pred_direction, count_pred_novelty):
        # get label with max votes and its count
        max_rel = max(rel_votes, key=rel_votes.get)
        max_rel_count = rel_votes[max_rel]
        max_pos_rel = ''

        max_dir = max(dir_votes, key=dir_votes.get)
        max_dir_count = dir_votes[max_dir]
        max_nov = max(nov_votes, key=nov_votes.get)
        max_nov_count = nov_votes[max_nov]
        if max_rel == 'None':
            max_nov = 'None'
            max_dir = 'None'
        #elif max_rel != 'Positive_Correlation' and max_rel != 'Negative_Correlation' and max_rel != 'Conversion':
        #    max_dir = 'None'
        if max_rel != 'None' and max_nov == 'None':
            max_nov = 'Novel'

        if max_rel != 'None' and max_dir == 'None':
            max_dir = 'No_Direct'
        if max_rel_count >= num_config / 2:
            ensemble_pred_labels.append((max_rel, max_nov, max_dir))
        else:
            ensemble_pred_labels.append(('None', 'None', 'None'))
    
    with open(out_tsv_file, 'w') as f:
        f.write('config\tpair_f1\tpair_novelty_f1\trelation_f1\trelation_novelty_f1\trelation_direction_f1\tall_direction_f1\n')
        for config, seed_scores in config_2_scores.items():
            average_scores = [s['f1'] for s in seed_scores[0]]
            for seed_score in seed_scores[1:]:
                for i in range(len(average_scores)):
                    #print(seed_score[i])
                    average_scores[i] += seed_score[i]['f1']
            average_scores = [score / len(seed_scores) for score in average_scores]
            f.write(config + '\t' + '\t'.join([str(score) for score in average_scores]) + '\n')

        ensemble_pair_pred_labels = [('Association', '_') if rel != 'None' else ('None', '_') for rel, nov, dir in ensemble_pred_labels]
        ensemble_pair_novelty_pred_labels = [('Association', nov) if rel != 'None' else ('None', '_') for rel, nov, dir in ensemble_pred_labels]
        ensemble_relation_pred_labels = [(rel, '_') if rel != 'None' else ('None', '_') for rel, nov, dir in ensemble_pred_labels]
        ensemble_relation_novelty_pred_labels = [(rel, nov) if rel != 'None' else ('None', '_') for rel, nov, dir in ensemble_pred_labels]
        ensemble_relation_direction_pred_labels = [(rel, dir) if rel != 'None' else ('None', '_') for rel, nov, dir in ensemble_pred_labels]
        pair_f1 = eval(ensemble_pair_pred_labels, load_gold_labels(in_gold_tsv_file, level = 'pair'))
        pair_novelty_f1 = eval(ensemble_pair_novelty_pred_labels, load_gold_labels(in_gold_tsv_file, level = 'pair+novelty'))
        relation_f1 = eval(ensemble_relation_pred_labels, load_gold_labels(in_gold_tsv_file, level = 'relation'))
        relation_novelty_f1 = eval(ensemble_relation_novelty_pred_labels, load_gold_labels(in_gold_tsv_file, level = 'relation+novelty'))
        relation_direction_f1 = eval(ensemble_relation_direction_pred_labels, load_gold_labels(in_gold_tsv_file, level = 'relation+direction'))
        all_direction_f1 = eval(ensemble_pred_labels, load_gold_labels(in_gold_tsv_file, level = 'all'))    
        f.write('Ensemble\t' + '\t'.join([str(score) for score in [pair_f1['f1'], pair_novelty_f1['f1'], relation_f1['f1'], relation_novelty_f1['f1'], relation_direction_f1['f1'], all_direction_f1['f1']]]) + '\n')
        print('Ensemble')
        print(relation_novelty_f1)
        print()
    

def run_eval_on_bc8(in_pred_dir):
    
    #in_gold_tsv_file         = 'datasets/ncbi_relation/processed/test.biorex2.direction.tsv'
    in_gold_tsv_file         = 'datasets/bioredirect/bioredirect_test.pubtator.tsv'
    in_pred_tsv_file_pattern = in_pred_dir + '/val_pred_*.tsv'
    out_tsv_file             = in_pred_dir + '_val_pred_summary_score.tsv'

    run_eval(in_gold_tsv_file,
             in_pred_tsv_file_pattern,
             out_tsv_file)
    
    out_tsv_file             = in_pred_dir + '_val_pred_summary_score2.tsv'

    run_eval(in_gold_tsv_file,
             in_pred_tsv_file_pattern,
             out_tsv_file,
             threshold = 0.5)

    #in_gold_tsv_file         = 'datasets/bc8/test.biorex2.direction.tsv'
    in_gold_tsv_file         = 'datasets/bioredirect/bioredirect_bc8_test.pubtator.tsv'
    in_pred_tsv_file_pattern = in_pred_dir + '/test_pred_*.tsv'
    out_tsv_file             = in_pred_dir + '_test_pred_summary_score.tsv'

    run_eval(in_gold_tsv_file,
             in_pred_tsv_file_pattern,
             out_tsv_file)
    
    out_tsv_file             = in_pred_dir + '_test_pred_summary_score2.tsv'
    
    run_eval(in_gold_tsv_file,
             in_pred_tsv_file_pattern,
             out_tsv_file,
             threshold = 0.5)

def run_eval_on_biored1(in_pred_dir):
    
    in_gold_tsv_file         = 'datasets/bioredirect/bioredirect_dev.pubtator.tsv'
    in_pred_tsv_file_pattern = in_pred_dir + '/val_pred_*.tsv'
    out_tsv_file             = in_pred_dir + '_val_pred_summary_score.tsv'

    run_eval(in_gold_tsv_file,
             in_pred_tsv_file_pattern,
             out_tsv_file)
    
    out_tsv_file             = in_pred_dir + '_val_pred_summary_score2.tsv'

    run_eval(in_gold_tsv_file,
             in_pred_tsv_file_pattern,
             out_tsv_file,
             threshold = 0.5)

    in_gold_tsv_file         = 'datasets/bioredirect/bioredirect_test.pubtator.tsv'
    in_pred_tsv_file_pattern = in_pred_dir + '/test_pred_*.tsv'
    out_tsv_file             = in_pred_dir + '_biored1_summary_score.tsv'

    run_eval(in_gold_tsv_file,
             in_pred_tsv_file_pattern,
             out_tsv_file)
    
    out_tsv_file             = in_pred_dir + '_biored1_summary_score2.tsv'
    
    run_eval(in_gold_tsv_file,
             in_pred_tsv_file_pattern,
             out_tsv_file,
             threshold = 0.5)

if __name__ == '__main__':

    run_eval_on_bc8(in_pred_dir = 'results_bioredirect_on_new_bc8_rel_best')
    #run_eval_on_bc8(in_pred_dir = 'results_bioredirect_on_new_bc8_pair_best')
    #run_eval_on_biored1(in_pred_dir = 'results_bioredirect_on_new_biored_test_infix')
    #run_eval_on_biored1(in_pred_dir = 'results_bioredirect_on_new_biored_test_prefix')
    #run_eval_on_biored1(in_pred_dir = 'results_bioredirect_on_new_biored_test_suffix')
    #run_eval_on_bc8(in_pred_dir = 'results_bioredirect_on_new_bc8_with_data_label_inconsist_loss_01_e50_fix_dir_bug')
    #run_eval_on_biored1(in_pred_dir = 'results_bioredirect_on_biored_with_data_label_inconsist_loss_01')
    #run_eval_on_bc8(in_pred_dir = 'results_bioredirect_on_new_bc8_suffix')
    #run_eval_on_bc8(in_pred_dir = 'results_bioredirect_on_new_bc8_prefix')
    #run_eval_on_bc8(in_pred_dir = 'results_bioredirect_on_new_bc8_infix')
    #run_eval_on_bc8(in_pred_dir = 'results_bioredirect_on_new_bc8_only_re_loss')
    #run_eval_on_bc8(in_pred_dir = 'results_bioredirect_on_new_bc8_with_data_label_inconsist_loss_01')
    #run_eval_on_bc8(in_pred_dir = 'results_bioredirect_on_bc8_weight_by_labels_num_e10_no_None_loss_attention_loss')
    #run_eval_on_bc8(in_pred_dir = 'results_bioredirect_on_bc8_weight_by_labels_num_e10_no_None_loss')
    #run_eval_on_bc8(in_pred_dir = 'results_bioredirect_on_bc8_weight_by_labels_num_e10_sigmod')
    #run_eval_on_bc8(in_pred_dir = 'results_bioredirect_on_bc8_weight_by_labels_num_e10')
    #run_eval_on_bc8(in_pred_dir = 'results_bioredirect_on_bc8_no_weight_e3+weight_e10+biorex2')
    #run_eval_on_bc8(in_pred_dir = 'results_bioredirect_on_bc8')
    #run_eval_on_biored1(in_pred_dir = 'results_bioredirect_on_biored1')
    #run_eval_on_bc8(in_pred_dir = 'results_biorex2')
    #run_eval2(in_pred_dir = 'results_no_ml_biorex2_infix')
    #run_eval2(in_pred_dir = 'results_no_ml_biorex2_suffix')
    #run_eval2(in_pred_dir = 'results_no_ml_biorex2_prefix')




