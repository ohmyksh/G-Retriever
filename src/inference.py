import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import json
import pandas as pd
from dataset.webqsp_pcst import WebQSP
from datasets import concatenate_datasets
import transformers
from model import GraphLLM


# def main():
#     # Load webqsp dataset
#     dataset = WebQSP()
#     dataset = concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
#     idx_split = dataset.get_idx_split()
#     eval_batch_size = 16 
    
#     test_dataset = [dataset[i] for i in idx_split['test']]
#     test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, drop_last=False, pin_memory=True, shuffle=False)
    
#     os.makedirs('/home/shkim/G-Retriever-Implement'+'/result/webqsp', exist_ok=True)
#     output_path = '/home/shkim/G-Retriever-Implement/result/webqpsp'
    
#     GraphLLM.eval()
#     progress_bar_test = tqdm(range(len(test_loader)))
    
#     with open(output_path, "w") as f:
#         # not using batch
#         for _, sample in enumerate(test_loader):
#             with torch.no_grad():
#                 output = GraphLLM.inference(sample)
#                 pred = pd.DataFrame(output)
#                 # for _, row in pred.iterrows():
#                 #     f.write(json.dumps(dict(row)) + "\n")
#             progress_bar_test.update(1)

def main():
    # Load webqsp dataset
    dataset = WebQSP()
    idx_split = dataset.get_idx_split()
    eval_batch_size = 1  # Batch size 1 for single sample testing
    
    test_dataset = [dataset[i] for i in idx_split['test']]
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, drop_last=False, pin_memory=True, shuffle=False)
    
    os.makedirs('/home/shkim/G-Retriever-Implement'+'/result/webqsp', exist_ok=True)
    output_path = '/home/shkim/G-Retriever-Implement/result/webqpsp'
    
    GraphLLM.eval()
    
    # Fetch the first sample only
    sample = next(iter(test_loader))
    
    with torch.no_grad():
        output = GraphLLM.inference(sample)
        pred = pd.DataFrame(output)
        print(pred)  # Print the output for the single sample

        # Save the output if needed
        with open(output_path, "w") as f:
            # If saving the result as JSON
            for _, row in pred.iterrows():
                f.write(json.dumps(dict(row)) + "\n")
                
main()