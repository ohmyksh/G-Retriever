# Paper Implementation
# 🔎 G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering (Arxiv 2024)   
📑  Paper: [(link)](https://arxiv.org/pdf/2402.07630)  
💾  Original Repo: [(link)]([https://github.com/IDEA-FinAI/ToG](https://github.com/XiaoxinHe/G-Retriever))  

## 🌱 Contents
1. [Paper summary](#Summary)
2. [Module design](#Modules)
3. [Implementation details](#Implementation-Details)
4. [Discussion and opportunities for improvement](#Discussion-and-Opportunities-for-improvement)


## Summary
The paper introduces a new GraphQA benchmark for real-world graph question answering and present G-Retriever, an architecture adept at complex and creative queries. Experimental results show that G-Retriever surpasses baselines in textual graph tasks across multiple domains, scales effectively with larger graph sizes, and demonstrates resistance to hallucination.

## Modules
<img src="figs/g-retriever-figure.png" alt="My Illustration for Implementation" width="800">
G-Retriever consists of four stages. First, it indexes the node and edge text attributes of a given textual graph to generate embeddings. Based on these embeddings, it constructs a subgraph composed of nodes and edges relevant to the given query. Finally, an LLM generates the final response using the subgraph and query as input prompts.
## Modules

1. **Indexing**
   - **Input**: Text attributes of nodes and edges, pre-trained LM (e.g., SentenceBERT)
   - **Output**: \( z_n \), \( z_e \)
   - Embeddings for node and edge text attributes are generated using a pre-trained model like SentenceBERT. 

2. **KNN-Retriever**
   - **Input**: Query text attribute, pre-trained LM
   - **Output**: Top-k most relevant nodes and edges
   - Using the same encoding approach as indexing, query embeddings are generated and the top-k most relevant nodes and edges are retrieved using k-nearest neighbors.

3. **Subgraph Constructor**
   - **Input**: Top-k most relevant nodes and edges
   - **Output**: \( S^* = (V^*, E^*) \)
   - The Prize-Collecting Steiner Tree optimization is used to find the connected subgraph \( S^* \) that maximizes node value while minimizing edge cost.

4. **Answer Generator**
   - **Input**: \( S^* = (V^*, E^*) \), query embedding
   - **Output**: Final answer \( Y \)
   - The final response is generated by an LLM using graph tokens, query embeddings, and submodules:
     1) **Graph Encoder**: Generates embedding tokens for \( S^* \) using mean pooling.
     2) **Projection Layer**: Aligns graph tokens with the LLM's vector space using an MLP.
     3) **Text Embedder**: Converts \( S^* \) into a textual format.
     4) **LLM Generation**: Produces the final answer \( Y \), using graph tokens as soft prompts.

## Implementation Details
* Dataset: ExplaGraphs
* Metric: Exact Match Accuracy (Hits@1)
* Model: Llama2-7b(LLM), SentenceBert(encoder), Graph Transformer(graph encoder)

## Discussion and Opportunities for improvement 
- Subgraphs may be constructed inaccurately. A subgraph is selected based on the total score calculated from the similarity between node/edge embeddings and the query. However, while the total score may be high, it could include irrelevant elements or fail to include highly relevant ones.
- Elements necessary for response generation may have low relevance to the query (e.g., multi-hop reasoning).
- The model may fail to fully leverage the graph's structural (topological) information. Since only individual embeddings for nodes and edges are generated, the connectivity within the graph is not adequately captured.
