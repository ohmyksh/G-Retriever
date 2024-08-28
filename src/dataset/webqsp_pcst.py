import os
import torch
import pandas as pd
from torch.utils.data import Dataset
import datasets
from tqdm import tqdm
from dataset.utils.pcst import pcst

# for webqsp dataset
path = '/mnt/nvme1n1p2/shkim/webqsp'
nodes_path = f'{path}/nodes'
edges_path = f'{path}/edges'
graphs_path = f'{path}/graphs'

subgraphs = f'{path}/subgraphs'
os.makedirs(subgraphs, exist_ok=True)
    
class WebQSP(Dataset):
    def __init__(self):
        super().__init__()
        self.prompt = 'Please answer the given question.'
        self.graph = None
        dataset = datasets.load_dataset("rmanluo/RoG-webqsp")
        self.dataset = datasets.concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
        print("Loading q_embs...")
        self.q_embs = torch.load(f'/mnt/nvme1n1p2/shkim/webqsp/q_embs.pt')
        print("load!!!!!")
        self.len = len(self.dataset)
    
    def __getitem__(self, i):
        data = self.dataset[i]
        question = f'Question: {data["question"]}\nAnswer: '
        graph = torch.load(f'{subgraphs}/{i}.pt')
        textualized = open(f'{subgraphs}/textualized/{i}.txt', 'r').read()
        # nodes = torch.load(f'{nodes_path}/{i}.csv')
        label = ('|').join(data['answer']).lower()
        
        return {
            'id': i,
            'question': question,
            'label': label,
            'graph': graph,
            'textualized': textualized
        }
        
    def get_idx_split(self):
        with open(f'{path}/train_indices.txt', 'r') as file:
            train_indices = [int(line.strip()) for line in file]
        with open(f'{path}/val_indices.txt', 'r') as file:
            val_indices = [int(line.strip()) for line in file]
        with open(f'{path}/test_indices.txt', 'r') as file:
            test_indices = [int(line.strip()) for line in file]

        return {'train': train_indices, 'val': val_indices, 'test': test_indices}
    
    
def subgraph_retrieval(dataset):
    q_embs = torch.load(f'{path}/q_embs.pt')
    for i in tqdm(range(dataset.len)):
        graph = torch.load(f'{graphs_path}/graph_{i}.pt')
        nodes = pd.read_csv(f'{nodes_path}/{i}.csv')
        edges = pd.read_csv(f'{edges_path}/{i}.csv')
        q_emb = q_embs[i]
        # subgraph retrieval
        # pcst parameter for webqsp
        topk_n = 3
        topk_e = 5
        cost_e = 0.5
        
        print(f"Processing subgraph retrieval for sample {i}...")
        subgraph, textualized = pcst(graph, q_emb, nodes, edges, topk_n, topk_e, cost_e)
        torch.save(subgraph, f'{subgraphs}/{i}.pt')
        os.makedirs(subgraphs+'/textualized', exist_ok=True)
        open(f'{subgraphs}/textualized/{i}.txt', 'w').write(textualized)


if __name__ == '__main__':
    dataset = WebQSP()
    subgraph_retrieval(dataset)
    