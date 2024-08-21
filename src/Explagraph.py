import json
import pandas as pd
import torch
from torch.utils.data import Dataset

# for explagraphs dataset
path = '/home/shkim/G-Retriever-Implement/data/expla_graphs'

class ExplaGraphs(Dataset):
    def __init__(self):
        super().__init__()
        self.text = pd.read_csv(f'{path}/train_dev.tsv', sep='\t')
        self.prompt = 'Question: Do argument 1 and argument 2 support or counter each other? Answer in one word in the form of \'support\' or \'counter\'.\n\nAnswer:'
        self.graph = None
        
    def __getitem__(self, index):
        text = self.text.iloc[index]
        graph = torch.load(f'{path}/graphs/graph_{index}.pt')
        question = f'Argument 1: {text.arg1}\nArgument 2: {text.arg2}\n{self.prompt}'
        
        return {
            'id': index,
            'label': text['label'],
            'graph': graph,
            'question': question,
        }

    def get_idx_split(self):
        with open(f'{path}/split/train_indices.txt', 'r') as file:
            train_indices = [int(line.strip()) for line in file]

        with open(f'{path}/split/val_indices.txt', 'r') as file:
            val_indices = [int(line.strip()) for line in file]

        with open(f'{path}/split/test_indices.txt', 'r') as file:
            test_indices = [int(line.strip()) for line in file]

        return {'train': train_indices, 'val': val_indices, 'test': test_indices}


# Test code
def test_get_item(dataset, index):
    item = dataset[index]
    print(f"Item at index {index}:")
    print(f"ID: {item['id']}")
    print(f"Label: {item['label']}")
    print(f"Graph: {item['graph']}")
    print(f"Question: {item['question']}\n")

dataset = ExplaGraphs()

# get item test
test_get_item(dataset, 0)  
test_get_item(dataset, 1)  