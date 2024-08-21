import pandas as pd
import re
import torch
import os
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from tqdm import tqdm

# for explagraphs dataset
path = '/home/shkim/G-Retriever-Implement/data/expla_graphs/'

def generate_split(num_nodes, path):

    # Split the dataset into train, val, and test sets
    indices = np.arange(num_nodes)
    train_indices, temp_data = train_test_split(indices, test_size=0.4, random_state=42)
    val_indices, test_indices = train_test_split(temp_data, test_size=0.5, random_state=42)
    print("# train samples: ", len(train_indices))
    print("# val samples: ", len(val_indices))
    print("# test samples: ", len(test_indices))

    # Create a folder for the split
    os.makedirs(path, exist_ok=True)

    # Save the indices to separate files
    with open(f'{path}/train_indices.txt', 'w') as file:
        file.write('\n'.join(map(str, train_indices)))

    with open(f'{path}/val_indices.txt', 'w') as file:
        file.write('\n'.join(map(str, val_indices)))

    with open(f'{path}/test_indices.txt', 'w') as file:
        file.write('\n'.join(map(str, test_indices)))
        

os.makedirs(path+'graphs', exist_ok=True)

# for explagraph.

# def explagraph_preprocess():
#     dataset_path = path + 'train_dev.tsv'
#     dataset = pd.read_csv(dataset_path, sep='\t')

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
    graphs = extract_graph(dataset)
    # create nodes, edges embeddings by using sentence bert
    
    print('Encoding graphs...')
    # Create node embeddings
    for i, graph in tqdm(enumerate(graphs), total=len(graphs), desc="Encoding Graphs"): 
        # Create node embeddings
        node_attributes = list(graph['nodes'].keys())
        node_embeddings = sbert.encode(node_attributes, convert_to_tensor=True)
        
        # Create edge embeddings
        edge_attributes = [edge['edge_attr'] for edge in graph['edges']]
        edge_embeddings = sbert.encode(edge_attributes, convert_to_tensor=True)
        edge_index = torch.LongTensor([[edge['src'], edge['dst']] for edge in graph['edges']])
        num_nodes = len(graph['nodes'])
        
        data = Data(node_embed=node_embeddings, edge_index=edge_index, edge_embed=edge_embeddings, num_nodes=num_nodes)
        torch.save(data, f'{path}/graphs/graph_{i}.pt')
        
        # print(f"Graph {i + 1} processed and saved.")
        # print(f" - Number of nodes: {num_nodes}")
        # print(f" - Node embeddings shape: {node_embeddings.shape}")
        # print(f" - Edge embeddings shape: {edge_embeddings.shape}")
        # print(f" - Edge index shape: {edge_index.shape}")
        # print(f" - Saved as: {path}/graph_{i}.pt")
        # print("------------------------------------------------\n")
    

if __name__ == '__main__':
    # Load dataset
    dataset_path = path + 'train_dev.tsv'
    dataset = pd.read_csv(dataset_path, sep='\t')
    
    # Indexing nodes, edges in graph
    graph_indexing(dataset)
    
    # Split the dataset into train, val, and test sets
    split_path = path + 'split'
    generate_split(len(dataset), split_path)
    
    