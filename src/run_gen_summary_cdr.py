import glob
import re

def load_pred_labels(in_pred_tsv_file,
                     threshold = 0.0):
    pred_labels = []
    relation_labels = ['None',
                       'CID']
    with open(in_pred_tsv_file, 'r') as f:
        f.readline()
        for line in f:
            #tks = line.rstrip().split('\t')
            tks = re.split(r'\s+', line.rstrip().strip(']').strip('['))
            rel_scores = []
            for i in range(0, len(relation_labels)):
                rel_scores.append(float(tks[i]))

            if threshold == 0:
                max_score = 0
                rel_label = ''
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

            pred_labels.append((rel_label))


    return pred_labels

def load_gold_labels(in_gold_tsv_file):

    gold_labels = []

    with open(in_gold_tsv_file, 'r') as f:
        for line in f:
            tks = line.rstrip().split('\t')
            rel_label = tks[6]
            id1       = tks[3]
            id2       = tks[4]
            gold_labels.append((rel_label))

    return gold_labels

def eval(pred_labels, gold_labels):
    if len(pred_labels) != len(gold_labels):
        return None

    tp = 0.
    fp = 0.
    fn = 0.
    for pred, gold in zip(pred_labels, gold_labels):
        if gold != 'None':
            if pred == gold:
                tp += 1
            elif pred == 'None':
                fn += 1
            else:
                fp += 1
                fn += 1
        elif pred != 'None':
            fp += 1
            
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall    = tp / (tp + fn) if tp + fn > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return { 'f1': f1, 'precision': precision, 'recall': recall, 'tp': tp, 'fp': fp, 'fn': fn }

def run_eval(in_gold_tsv_file,
             in_pred_tsv_file_pattern,
             out_tsv_file,
             threshold = 0.0):

    config_2_scores = {}
    for in_pred_tsv_file in glob.glob(in_pred_tsv_file_pattern):
        #print(in_pred_tsv_file)
        _, _, seed, config = in_pred_tsv_file.split('\\')[-1].split('_', 3)
        print(seed, config)
        
        pred_labels = load_pred_labels(in_pred_tsv_file, threshold = threshold)
        gold_labels = load_gold_labels(in_gold_tsv_file)
        f1 = eval(pred_labels, gold_labels)
        if f1 is not None:    
            config = config.replace('.tsv', '')
            if config not in config_2_scores:
                config_2_scores[config] = []
            print(f1)
            config_2_scores[config].append((f1['precision'], f1['recall'], f1['f1']))
            #print(f1)
    
    with open(out_tsv_file, 'w') as f:
        f.write('config\tprecision\trecall\tf1\n')
        for config, seed_scores in config_2_scores.items():

            prf_scores = list(seed_scores[0])
            num_seed = len(seed_scores)
            for _seed_score in seed_scores[1:]:
                for i in range(len(prf_scores)):
                    prf_scores[i] += _seed_score[i]
            average_scores = [score / num_seed for score in prf_scores]
            f.write(config + '\t' + '\t'.join([str(score) for score in average_scores]) + '\n')
    

def run_eval_old(in_pred_dir):
    
    #in_gold_tsv_file         = 'datasets/cdr/processed/dev.tsv'
    #in_gold_tsv_file         = 'datasets/cdr/processed/dev200.tsv'
    in_gold_tsv_file         = 'datasets/cdr/processed/old/train_dev.tsv'
    in_pred_tsv_file_pattern = in_pred_dir + '/val_pred_*.tsv'
    out_tsv_file             = in_pred_dir + '_val_pred_summary_score.tsv'

    run_eval(in_gold_tsv_file,
             in_pred_tsv_file_pattern,
             out_tsv_file)

    in_gold_tsv_file         = 'datasets/cdr/processed/old/test.tsv'
    in_pred_tsv_file_pattern = in_pred_dir + '/test_pred_*.tsv'
    out_tsv_file             = in_pred_dir + '_test_pred_summary_score.tsv'

    run_eval(in_gold_tsv_file,
             in_pred_tsv_file_pattern,
             out_tsv_file)
    
def run_eval_new(in_pred_dir):
    
    #in_gold_tsv_file         = 'datasets/cdr/processed/dev.tsv'
    #in_gold_tsv_file         = 'datasets/cdr/processed/dev200.tsv'
    in_gold_tsv_file         = 'datasets/cdr/processed/train_dev.tsv'
    in_pred_tsv_file_pattern = in_pred_dir + '/val_pred_*.tsv'
    out_tsv_file             = in_pred_dir + '_val_pred_summary_score.tsv'

    run_eval(in_gold_tsv_file,
             in_pred_tsv_file_pattern,
             out_tsv_file)

    in_gold_tsv_file         = 'datasets/cdr/processed/test.tsv'
    in_pred_tsv_file_pattern = in_pred_dir + '/test_pred_*.tsv'
    out_tsv_file             = in_pred_dir + '_test_pred_summary_score.tsv'

    run_eval(in_gold_tsv_file,
             in_pred_tsv_file_pattern,
             out_tsv_file)
    
def run_eval_new_val_on_dev(in_pred_dir):
    
    in_gold_tsv_file         = 'datasets/cdr/processed/dev.tsv'
    in_pred_tsv_file_pattern = in_pred_dir + '/val_pred_*.tsv'
    out_tsv_file             = in_pred_dir + '_val_pred_summary_score.tsv'

    run_eval(in_gold_tsv_file,
             in_pred_tsv_file_pattern,
             out_tsv_file)

    in_gold_tsv_file         = 'datasets/cdr/processed/test.tsv'
    in_pred_tsv_file_pattern = in_pred_dir + '/test_pred_*.tsv'
    out_tsv_file             = in_pred_dir + '_test_pred_summary_score.tsv'

    run_eval(in_gold_tsv_file,
             in_pred_tsv_file_pattern,
             out_tsv_file)

if __name__ == '__main__':

    #run_eval_old(in_pred_dir = 'results_cdr_train_all') # prompt is not tokenized
    #run_eval_new(in_pred_dir = 'results_bioredirect_on_new_cdr') # prompt is  tokenized
    #run_eval_new_val_on_dev(in_pred_dir = 'results_bioredirect_on_new_cdr_val_on_dev_e50') # prompt is  tokenized
    run_eval_new_val_on_dev(in_pred_dir = 'results_bioredirect_on_new_cdr_val_on_train_dev_e50') # prompt is  tokenized




