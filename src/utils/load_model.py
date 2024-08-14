from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn

sbert_path = 'sentence-transformers/all-roberta-large-v1'