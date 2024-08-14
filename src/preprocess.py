import pandas as pd
import re
import torch
import os
from sentence_transformers import SentenceTransformer

# for explagraph.

# def explagraph_preprocess():
#     dataset_path = '/home/shkim/G-Retriever-Implement/data/expla_graphs/train_dev.tsv'
#     dataset = pd.read_csv(dataset_path, sep='\t')
#     embeddings = graph_indexing(dataset)
    
# def graph_indexing(dataset):
#     sbert = SentenceTransformer('sentence-transformers/all-roberta-large-v1')
    
#     all_node_embeddings = []
#     all_edge_embeddings = []
    
#     # extract nodes, edges
#     for index, row in dataset.itterrows():
#         print(f"Processing row {index + 1}/{len(dataset)}")
        
#         graph = row['graph']
#         triplets = re.findall(r'\((.*?)\)', graph)
#         nodes = {}
#         edges = []
#         for t in triplets:
#             src, edge, dst = t.split(';')
#             src = src.lower().strip()
#             dst = dst.lower().strip()
#             edge, edge.lower().strip()
#             if src not in nodes:
#                 nodes[src] = len(nodes)
#             if dst not in nodes:
#                 nodes[dst] = len(nodes)
#             edges.append({'src': src, 'edge': edge, 'dst': dst})
            
#         print(f" - Nodes extracted: {list(nodes.keys())}")
#         print(f" - Edges extracted: {edges}")
        
    
#         # create nodes, edges embeddings by using sentence bert
#         print('Encoding graphs...')
        
#         # Create node embeddings
#         node_embeddings = {}
#         for node in nodes.keys():
#             embedding = sbert.encode(node)
#             node_embeddings[node] = embedding
#         all_node_embeddings.append(node_embeddings)
        
#         print(f" - Node embeddings created for row {index + 1}")
        
#         # Create edge embeddings
#         edge_embeddings = []
#         for edge in edges:
#             edge_text = f"{edge['src']} {edge['edge']} {edge['dst']}"
#             embedding = sbert.encode(edge_text)
#             edge_embeddings.append(embedding)
#         all_edge_embeddings.append(edge_embeddings)
        
#         print(f" - Edge embeddings created for row {index + 1}")
    
#     print("All rows processed.")
#     return [all_node_embeddings, all_edge_embeddings]
                
    
    

def explagraph_preprocess():
    dataset_path = '/home/shkim/G-Retriever-Implement/data/expla_graphs/train_dev.tsv'
    dataset = pd.read_csv(dataset_path, sep='\t')
    # 테스트를 위해 첫 번째 행만 전달합니다.
    embeddings = graph_indexing(dataset.head(1))  
    print('Single row embedding generated successfully.')

def graph_indexing(dataset):
    sbert = SentenceTransformer('sentence-transformers/all-roberta-large-v1')
    
    all_node_embeddings = []
    all_edge_embeddings = []
    
    # Extract nodes, edges from the first row only
    for index, row in dataset.iterrows():
        print(f"Processing row {index + 1}/{len(dataset)}")
        
        graph = row['graph']
        triplets = re.findall(r'\((.*?)\)', graph)
        nodes = {}
        edges = []
        for t in triplets:
            src, edge, dst = t.split(';')
            src = src.lower().strip()
            dst = dst.lower().strip()
            edge = edge.lower().strip()
            if src not in nodes:
                nodes[src] = len(nodes)
            if dst not in nodes:
                nodes[dst] = len(nodes)
            edges.append({'src': src, 'edge': edge, 'dst': dst})
            
        print(f" - Nodes extracted: {list(nodes.keys())}")
        print(f" - Edges extracted: {edges}")
        
        # Create node embeddings
        print('Encoding nodes...')
        node_embeddings = {}
        for node in nodes.keys():
            embedding = sbert.encode(node)
            node_embeddings[node] = embedding
            print(f"   Node: {node} -> Embedding: {embedding}")
        all_node_embeddings.append(node_embeddings)
        
        print(f" - Node embeddings created for row {index + 1}")
        
        # Create edge embeddings
        print('Encoding edges...')
        edge_embeddings = []
        for edge in edges:
            edge_text = f"{edge['src']} {edge['edge']} {edge['dst']}"
            embedding = sbert.encode(edge_text)
            edge_embeddings.append(embedding)
            print(f"   Edge: {edge_text} -> Embedding: {embedding}")
        all_edge_embeddings.append(edge_embeddings)
        
        print(f" - Edge embeddings created for row {index + 1}")
    
    print("Single row processed.")
    return [all_node_embeddings, all_edge_embeddings]

# Example usage:
explagraph_preprocess()