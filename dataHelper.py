from datasets import load_dataset, DatasetDict, concatenate_datasets
import json
import random
from datasets import DatasetDict, Dataset
from datasets import ClassLabel, Value
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import os
import argparse
from datasets import load_from_disk

def stratified_sample(labels, n_samples, label_map):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=n_samples, random_state=0)
    for train_index, test_index in sss.split(np.zeros(len(labels)), labels):
        sampled_idx = test_index
        break
    return sampled_idx

# Datasets:
# rotten_tomatoes
# agnews
# go_emotions
# dair-ai/emotion

FEW_SHOT_NUMS = 16
# get current dir
PROJ_DIR = os.getcwd()
DATA_DIR = os.path.join(PROJ_DIR, 'data')
print(PROJ_DIR)

def get_dataset(dataset_name, sep_token = '<sep>'):
    '''
    dataset_name: str or list, the name of the dataset(s)
    sep_token: str, the sep_token used by tokenizer(e.g. '<sep>')
    '''
    print("-------------- DATASET loading.... --------------")
    if isinstance(dataset_name, str):
        dataset_name = [dataset_name]

    all_datasets = []

    for name in dataset_name:
        if name in ['rotten_tomatoes', 'ag_news', 'go_emotions', 'dair-ai/emotion']:
            dataset = load_train_dataset(name, sep_token)
        else:
            raise ValueError(f"Unknown dataset name: {name}")
        all_datasets.append(dataset)

    if len(all_datasets) == 1:
        return all_datasets[0]

    # For multi-dataset case, concatenate and re-label
    result = aggregate_datasets(all_datasets)
    print("-------------- DATASET loaded.... --------------")
    return result

def load_train_dataset(dataset_name, sep_token):
    train_dataset = load_dataset(dataset_name, split='test')
    # Split the dataset into training and test sets using the 9:1 ratio
    split_dataset = train_dataset.train_test_split(test_size=0.1, seed=2022)
    train_ds = split_dataset['train']
    test_ds = split_dataset['test']
    
    print(f"{dataset_name} loaded.")
    return DatasetDict({
        'train': train_ds,
        'test': test_ds,
    })


def aggregate_datasets(datasets):
    # Initialize lists to hold the aggregated data and labels
    aggregated_train_texts = []
    aggregated_train_labels = []
    aggregated_test_texts = []
    aggregated_test_labels = []
    
    # We start with a base label which will be incremented to avoid label overlapping
    base_label = 0
    
    for dataset in datasets:
        # Update the train dataset labels and extend the aggregated lists
        train_labels = [label + base_label for label in dataset['train']['label']]
        aggregated_train_texts.extend(dataset['train']['text'])
        aggregated_train_labels.extend(train_labels)
        
        # Update the test dataset labels and extend the aggregated lists
        test_labels = [label + base_label for label in dataset['test']['label']]
        aggregated_test_texts.extend(dataset['test']['text'])
        aggregated_test_labels.extend(test_labels)
        
        # Update the base_label for the next dataset to avoid label overlapping
        # Assuming each task has the same number of labels
        unique_labels = set(train_labels + test_labels)
        base_label += len(unique_labels)
    
    # Create the final aggregated datasets
    aggregated_train_dataset = Dataset.from_dict({'text': aggregated_train_texts, 
                                                  'label': aggregated_train_labels})
    aggregated_test_dataset = Dataset.from_dict({'text': aggregated_test_texts, 
                                                 'label': aggregated_test_labels})
    
    # Return as a DatasetDict
    return DatasetDict({
        'train': aggregated_train_dataset,
        'test': aggregated_test_dataset
    })

def dbg(ds):
    return [entry['label'] for entry in ds]

if __name__ == "__main__":
    sep_token = '<sep>'
    dataset = get_dataset(['ag_news'], sep_token)