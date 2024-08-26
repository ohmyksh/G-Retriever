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
        
    def forward(self, x, adj_t, edge_attr):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index=adj_t, edge_attr=edge_attr)
            x = self.bns()[i](x)
            x = nn.ReLU(x)
            x = nn.Dropout(x, p=self.dropout, training=self.training)
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
        
    @property
    def device(self):
        return list(self.parameters())[0].device
    
    def graph_encoder(self, graph):
        graph = graph.to(self.model.device)
        # graph_encoder 
        self.graph_encoder = GraphTransformer(
            in_channels=1024,
            out_channels=1024,
            hidden_channels=1024,
            num_layers=4,
            dropout=0.0,
            num_heads=4,
        ).to(self.device)
        
        gnn_embedding = self.graph_encoder(graph.x, graph.edge_index.long(), graph.edge_attr)
        graph_embedding = scatter(gnn_embedding, dim=0, reduce='mean')
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
        
        question = self.tokenizer(sample["question"])
        # 1. Graph Encoder
        graph_embedding = self.graph_encoder(sample["graph"])
        
        # 2. Projection Layer
        projected_graph = self.projection(graph_embedding)
        
        # 3. Text Embedder
        textualized_graph = self.tokenizer(sample["textualized"])
        textembed_input = torch.concat([textualized_graph, question], dim=0)
        text_embedding = self.model.model.get_input_embeddings(torch.tensor(textembed_input).to(self.model.device))
        input_embedding = torch.cat([graph_embedding, text_embedding], dim=0)
        
        # 4. LLM Generation with Graph Prompt Tuning
        output = self.model.generate(
             inputs_embeds=input_embedding,
        )
        answer = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return answer