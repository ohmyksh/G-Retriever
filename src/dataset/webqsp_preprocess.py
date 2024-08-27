import pandas as pd
import re
import torch
import os
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm

# for webqsp dataset
path = '/home/shkim/G-Retriever-Implement/data/webqsp'
output_path = '/mnt/nvme1n1p2/shkim/webqsp'

def generate_split(num_nodes, path):

    # Split the dataset into train, val, and test sets
    indices = np.arange(num_nodes)
    train_indices, temp_data = train_test_split(indices, test_size=0.4, random_state=42)
    val_indices, test_indices = train_test_split(temp_data, test_size=0.5, random_state=42)
    print("# train samples: ", len(train_indices))
    print("# val samples: ", len(val_indices))
    print("# test samples: ", len(test_indices))

    # Create a folder for the split
    os.makedirs(output_path, exist_ok=True)

    # Save the indices to separate files
    with open(f'{output_path}/train_indices.txt', 'w') as file:
        file.write('\n'.join(map(str, train_indices)))

    with open(f'{output_path}/val_indices.txt', 'w') as file:
        file.write('\n'.join(map(str, val_indices)))

    with open(f'{output_path}/test_indices.txt', 'w') as file:
        file.write('\n'.join(map(str, test_indices)))
        

os.makedirs(output_path+'/graphs', exist_ok=True)

def extract_graph(dataset):
    graphs = []
    for index, row in dataset.iterrows():
        # print(f"Processing row {index + 1}/{len(dataset)}")
        graph = row['graph']
        triplets = re.findall(r'\((.*?)\)', graph)
        nodes = {}
        edges = []
        for t in triplets:
            src, edge, dst = t.split(';')
            src = src.lower().strip()
            dst = dst.lower().strip()
            edge, edge.lower().strip()
            if src not in nodes:
                nodes[src] = len(nodes)
            if dst not in nodes:
                nodes[dst] = len(nodes)
            edges.append({'src': nodes[src], 'edge_attr': edge, 'dst': nodes[dst]})
        graphs.append({
            'nodes': nodes,
            'edges': edges
        })
    return graphs

def graph_indexing(dataset):
    sbert = SentenceTransformer('sentence-transformers/all-roberta-large-v1')
    
    # extract nodes, edges
    for i in tqdm(range(len(dataset))):
        nodes = {}
        edges = []
        for tri in dataset[i]['graph']:
            h, r, t = tri
            h = h.lower()
            t = t.lower()
            if h not in nodes:
                nodes[h] = len(nodes)
            if t not in nodes:
                nodes[t] = len(nodes)
            edges.append({'src': nodes[h], 'edge_attr': r, 'dst': nodes[t]})
        nodes = pd.DataFrame([{'node_id': v, 'node_attr': k} for k, v in nodes.items()], columns=['node_id', 'node_attr'])
        edges = pd.DataFrame(edges, columns=['src', 'edge_attr', 'dst'])

        os.makedirs(output_path+'/nodes', exist_ok=True)
        os.makedirs(output_path+'/edges', exist_ok=True)
        
        nodes.to_csv(f'{output_path}/nodes/{i}.csv', index=False)
        edges.to_csv(f'{output_path}/edges/{i}.csv', index=False)
    
    print('Encoding graphs...')
    # Create node embeddings
    start_index = 3242
    
    for i in tqdm(range(start_index, len(dataset))):
        # Create node embeddings
        nodes = pd.read_csv(f'{output_path}/nodes/{i}.csv')
        nodes.node_attr.fillna("", inplace=True)
        node_attributes = nodes.node_attr.tolist()
        ## In paper code, fillna("", inplace=True) step exists.
        node_embeddings = sbert.encode(node_attributes, convert_to_tensor=True)
        
        # Create edge embeddings
        edges = pd.read_csv(f'{output_path}/edges/{i}.csv')
        edge_attributes = edges.edge_attr.tolist()
        edge_embeddings = sbert.encode(edge_attributes, convert_to_tensor=True)
        edge_index = torch.LongTensor([edges.src.tolist(), edges.dst.tolist()])
        num_nodes = len(nodes)
        
        data = Data(node_embed=node_embeddings, edge_index=edge_index, edge_embed=edge_embeddings, num_nodes=num_nodes)
        torch.save(data, f'{output_path}/graphs/graph_{i}.pt')
        
        # print(f"Graph {i + 1} processed and saved.")
        # print(f" - Number of nodes: {num_nodes}")
        # print(f" - Node embeddings shape: {node_embeddings.shape}")
        # print(f" - Edge embeddings shape: {edge_embeddings.shape}")
        # print(f" - Edge index shape: {edge_index.shape}")
        # print(f" - Saved as: {path}/graph_{i}.pt")
        # print("------------------------------------------------\n")
    
    print('Encoding question...')
    questions = [i['question'] for i in dataset]
    q_embs = sbert.encode(questions, convert_to_tensor=True)
    torch.save(q_embs, f'{output_path}/q_embs.pt')
    
    
if __name__ == '__main__':
    # Load dataset
    dataset = load_dataset("rmanluo/RoG-webqsp")
    dataset = concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
    
    # # Indexing nodes, edges in graph
    # graph_indexing(dataset)
    
    # Split the dataset into train, val, and test sets
    split_path = path + 'split'
    generate_split(len(dataset), split_path)
    
    