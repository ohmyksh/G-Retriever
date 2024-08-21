import pandas as pd
import re
import torch
import os
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# for explagraphs dataset
path = '/home/shkim/G-Retriever-Implement/data/expla_graphs/'

def encoding_question(dataset):
    
if __name__ == '__main__':
    # Load dataset
    dataset_path = path + 'train_dev.tsv'
    dataset = pd.read_csv(dataset_path, sep='\t')
    encoding_question(dataset)
    