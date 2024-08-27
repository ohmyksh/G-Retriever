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


def main():
    # Load webqsp dataset
    print("Loading WebQSP dataset...")
    dataset = WebQSP()
    idx_split = dataset.get_idx_split()
    
    eval_batch_size = 16 
    print("Preparing test dataset...")
    test_dataset = [dataset[i] for i in idx_split['test']]
    # test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, drop_last=False, pin_memory=True, shuffle=False)
    
    output_path = '/home/shkim/G-Retriever-Implement/result/webqpsp'
    os.makedirs(output_path, exist_ok=True)
    output_file = f'{output_path}/webqsp_result_0827.csv'


    model = GraphLLM()
    model.eval()
    progress_bar_test = tqdm(range(len(test_dataset)))
    
    with open(output_file, "w") as f:
        # not using batch
        print("Starting inference...")  # 추론 시작
        for i, sample in enumerate(test_dataset):
            print(f"Processing sample {i + 1}/{len(test_dataset)}") 
            with torch.no_grad():
                output = model.inference(sample)
                pred = pd.DataFrame(output)
                print(f"Sample {i + 1} inference complete.")
                for _, row in pred.iterrows():
                    f.write(json.dumps(dict(row)) + "\n")
            progress_bar_test.update(1)
        print("Inference complete. Results saved.")

            
if __name__ == "__main__":
    main()

            