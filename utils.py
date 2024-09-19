import torch
import numpy as np
import pandas as pd
import os
import glob
import csv

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_result_to_csv(result, filename):
    file_exists = os.path.isfile(filename)
    
    fieldnames = ['Dataset', 'Model', 'Pause Tokens', 'Trainer', 'Preprocessing', 'Training Time (s)', 'exact_match', 'f1', 'accuracy']

    with open(filename, 'a', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        if not file_exists or os.stat(filename).st_size == 0:
            dict_writer.writeheader()
        dict_writer.writerow(result)

def check_if_already_evaluated(results_file, dataset_name, model_name, num_pause_tokens, trainer_type):
    if not os.path.exists(results_file) or os.stat(results_file).st_size == 0:
        return False

    results_df = pd.read_csv(results_file)
    if 'Trainer' not in results_df.columns:
        return False

    already_evaluated = ((results_df['Dataset'] == dataset_name) &
                         (results_df['Model'] == model_name) &
                         (results_df['Pause Tokens'] == num_pause_tokens) &
                         (results_df['Trainer'] == trainer_type)).any()
    
    return already_evaluated

def load_latest_checkpoint(model_output_dir):
    checkpoint_dirs = glob.glob(os.path.join(model_output_dir, 'checkpoint-*'))
    if not checkpoint_dirs:
        return None
    latest_checkpoint_dir = max(checkpoint_dirs, key=os.path.getctime)
    return latest_checkpoint_dir
