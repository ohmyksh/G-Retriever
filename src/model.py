import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch_scatter import scatter
from torch_geometric.nn import TransformerConv

class GraphTransformer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_heads=-1):
        super(GraphTransformer, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(TransformerConv(in_channels=in_channels, out_channels=hidden_channels//num_heads, heads=num_heads, edge_dim=in_channels, dropout=dropout))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(TransformerConv(in_channels=hidden_channels, out_channels=hidden_channels//num_heads, heads=num_heads, edge_dim=in_channels, dropout=dropout,))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(TransformerConv(in_channels=hidden_channels, out_channels=out_channels//num_heads, heads=num_heads, edge_dim=in_channels, dropout=dropout,))
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.dropout_prob = dropout
        
        
    def forward(self, x, adj_t, edge_attr):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index=adj_t, edge_attr=edge_attr)
            x = self.bns[i](x)
            x = self.relu(x)
            x = self.dropout(x)
        x = self.convs[-1](x, edge_index=adj_t, edge_attr=edge_attr)
        return x, edge_attr


class GraphLLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        model_info = 'meta-llama/Llama-2-7b-hf'
        model = AutoModelForCausalLM.from_pretrained(
            model_info, 
            device_map="auto", 
            torch_dtype=torch.float16
            ).to("cuda")
        # freezing Llama
        for param in model.parameters():
            param.requires_grad = False
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(model_info)
        self.graph_encoder = GraphTransformer(
            in_channels=1024,
            out_channels=1024,
            hidden_channels=1024,
            num_layers=4,
            dropout=0.0,
            num_heads=4,
        ).to(self.device)
        
    @property
    def device(self):
        device = torch.device("cuda:0") 
        return torch.device("cuda:0")
    
    def graph_encoding(self, graph):
        graph = graph.to(self.model.device)
        
        # graph transformer
        gnn_embedding, _ = self.graph_encoder(
            graph.node_embed, 
            graph.edge_index.long(), 
            graph.edge_embed
            )
        
        # mean pooling
        graph_embedding = torch.mean(gnn_embedding, dim=0)
        
        print(f"Final graph embedding shape: {graph_embedding.shape}")

        print("Graph encoding completed.")
    
        return graph_embedding
    
    def projection(self, embed):
        projection = nn.Sequential(
                nn.Linear(1024, 2048),
                nn.ReLU(),
                nn.Linear(2048, 4096)
            ).to(self.model.device)
        return projection(embed)
    
        
    def inference(self, sample):
        #####################
        # Answer Generation
        #####################
        device = torch.device("cuda:0")
        print("Start Inference...")
        
        question = self.tokenizer(sample["question"],return_tensors="pt")
        print(f"Question: {question}")
        
        # 1. Graph Encoder
        print("Encoding graph...")
        graph_embedding = self.graph_encoding(sample["graph"])
        
        # 2. Projection Layer
        print("Projecting graph embedding...")
        projected_graph = self.projection(graph_embedding)
        
        # 3. Text Embedder
        print("Textualized graph...")
        textualized_graph = self.tokenizer(sample["textualized"],return_tensors="pt", add_special_tokens=False)
        textualized_graph = textualized_graph.input_ids.squeeze(0)
        question = question.input_ids.squeeze(0)
        
        textembed_input = torch.cat([textualized_graph[:512], question], dim=0)
        text_embedding = self.model.model.get_input_embeddings()(torch.tensor(textembed_input).to(self.model.device))
        
        # print(f"Projected Graph Embedding Shape: {projected_graph.shape}")
        # print(f"Text Embedding Shape: {text_embedding.shape}")

        # Ensure both tensors have the same number of dimensions
        if projected_graph.dim() == 1:
            projected_graph = projected_graph.unsqueeze(0)  # Convert to 2D tensor

        
        input_embedding = torch.cat([projected_graph, text_embedding], dim=0)
        print(f"Input Embedding: {input_embedding.shape}")
        
        print(f"Model device: {next(self.model.parameters()).device}")
        print(f"Graph embedding device: {graph_embedding.device}")
        print(f"Projected graph device: {projected_graph.device}")
        print(f"Text embedding device: {text_embedding.device}")
        print(f"Input embedding device: {input_embedding.device}")
        
        # 4. LLM Generation with Graph Prompt Tuning
        print("Generating answer using LLM...")
        output = self.model(
             inputs_embeds=input_embedding,
        )
        answer = self.tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Generated Answer: {answer}")
        return answer