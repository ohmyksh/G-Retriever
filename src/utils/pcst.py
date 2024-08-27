from pcst_fast import pcst_fast
import torch
import numpy as np
from torch_geometric.data import Data

def pcst(graph, q_emb, input_nodes, input_edges, topk_n, topk_e, cost_e):
    
    ##############################################
    # Step1: graph processing for pcst algorithm.
    # assign node prize, edge cost
    ##############################################
    #device = torch.device('cuda')
    graph = graph.cpu()
    q_emb = q_emb.cpu()
    
    # 1) Assign node prizes
    node_prizes = torch.nn.CosineSimilarity(dim=-1)(q_emb, graph.node_embed)
    if topk_n > 0:
        topk_n = min(topk_n, graph.num_nodes)
        _, topk_n_indices = torch.topk(node_prizes, topk_n, largest=True)
        
        # print("node_prizes device:", node_prizes.device)
        # print("topk_n_indices device:", topk_n_indices.device)
        # print("arange tensor device:", torch.arange(topk_n, 0, -1).device)
        
        node_prizes = torch.zeros_like(node_prizes)
        node_prizes[topk_n_indices] = torch.arange(topk_n, 0, -1).float()
        # k, k-1, ..., 1
    else:
        node_prizes = torch.zeros(graph.num_nodes)
        
    # 2) Assign edge prizes
    
    # In the paper, it is written that "Edge prizes are assigned similarly to node prizes," 
    # so I referred to the code for accurate implementation.
    
    c = 0.01  # Small constant for adjusting edge costs
    edge_prizes = torch.nn.functional.cosine_similarity(q_emb.unsqueeze(0), graph.edge_embed, dim=-1)
    if topk_e > 0:
        topk_e_values, _ = torch.topk(edge_prizes.unique(), topk_e, largest=True)
        edge_prizes[edge_prizes < topk_e_values[-1]] = 0.0
        last_topk_e_value = topk_e
        for k in range(topk_e):
            indices = edge_prizes == topk_e_values[k]
            value = min((topk_e-k)/sum(indices), last_topk_e_value)
            edge_prizes[indices] = value
            last_topk_e_value = value * (1 - c)
        # reduce the cost of the edges
        cost_e = min(cost_e, edge_prizes.max().item() * (1 - c/2))
    else:
        edge_prizes = torch.zeros(len(graph.edges))
    
    
    # 3) Calculate the cost of edges
    
    # create virtual nodes, virtual edges and make virtual graph to run pcst algorithm
    prizes = []
    costs = []
    edges = []
    virtual_node_prizes = []
    virtual_edges = []
    virtual_edges_cost = []
    virtual_node_to_edge = {}
    edge_to_original_index = {}
    
    for i, (src, dst) in enumerate(graph.edge_index.T.numpy()):
        edge_prize = edge_prizes[i]
        edge_cost = cost_e #predefined cost per edge
        
        if edge_prize <= edge_cost:
            reduced_cost = edge_cost - edge_prize
            edge_to_original_index[len(edges)] = i
            edges.append((src, dst))
            costs.append(reduced_cost)
        else: #need to create virtual node, since negative edge cost not allowed.
            virtual_node = graph.num_nodes + len(virtual_node_prizes)
            virtual_node_to_edge[virtual_node] = i
            virtual_edges.append((src, virtual_node))
            virtual_edges.append((virtual_node, dst))    
            virtual_edges_cost.extend([0,0])
            virtual_node_prizes.append(edge_prize-edge_cost) #negative prize
    
    # virtual graph
    prizes = np.concatenate([node_prizes, np.array(virtual_node_prizes)])
    edge_num = len(edges)
    costs = np.array(costs+virtual_edges_cost)
    edges = np.array(edges+virtual_edges)
    
    ######################################
    # Step2: run pcst and obtain subgraph
    ######################################
    root = -1
    num_clusters = 1
    pruning = 'gw'
    verbosity_level = 0
    
    n, e = pcst_fast(edges, prizes, costs, root, num_clusters, pruning, verbosity_level)
    
    ################################
    # Step3: recover original graph 
    ################################
    subgraph_nodes = n[n<graph.num_nodes]
    subgraph_edges = [edge_to_original_index[i] for i in e if i<edge_num]
    selected_virtual_nodes = n[n>=graph.num_nodes]
    if len(selected_virtual_nodes)>0:
        replaced_edges = [virtual_node_to_edge[i] for i in selected_virtual_nodes]
        subgraph_edges = np.array(subgraph_edges+replaced_edges)
    
    # select node based on edges in subgraph
    edge_index = graph.edge_index[:, subgraph_edges]
    # combine all nodes with src, dst nodes of edges
    subgraph_nodes = np.concatenate([subgraph_nodes, edge_index[0].numpy(), edge_index[1].numpy()])
    subgraph_nodes = np.unique(subgraph_nodes)
    
    node = input_nodes.iloc[subgraph_nodes]
    edge = input_edges.iloc[subgraph_edges]
    # node = [input_nodes[i] for i in subgraph_nodes]
    # edge = [input_edges[i] for i in subgraph_edges]
    mapping = {n: i for i, n in enumerate(subgraph_nodes.tolist())}
    
    # create subgraph data
    node_embedding = graph.node_embed[subgraph_nodes]
    edge_embed = graph.edge_embed[subgraph_edges]
    src = [mapping[i] for i in edge_index[0].tolist()]
    dst = [mapping[i] for i in edge_index[1].tolist()]
    edge_index = torch.LongTensor([src, dst])
    subgraph = Data(node_embed=node_embedding, edge_index=edge_index, edge_embed=edge_embed, num_nodes=len(subgraph_nodes))
    textualized = node.to_csv(index=False)+'\n'+edge.to_csv(index=False, columns=['src', 'edge_attr', 'dst'])
    
    return subgraph, textualized