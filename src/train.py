from model import GraphLLM
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import transformers
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
from dataset.webqsp_pcst import WebQSP
import wandb


# train configs
num_epochs=10
batch_size = 4
weight_decay = 0.05
initial_lr = 1e-5 # learning rate decays with a half-cycle cosine decay after the warm-up
patience = 2 # early stopping mechanisms

# train dataset
dataset = WebQSP()
idx_split = dataset.get_idx_split()
train_dataset = [dataset[i] for i in idx_split['train']]
dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# model
model = GraphLLM()
params = [p for _, p in model.named_parameters() if p.requires_grad]
    
# optimizer
optimizer = optim.AdamW(
    params, 
    lr=initial_lr,
    weight_decay=weight_decay,
    betas = (0.9, 0.95)
    )

# lr scheduler: learning rate decays with a half-cycle cosine decay after the warm-up period.
def lr_scheduler(optimizer, base_lr, current_step, total_steps):
    lr = base_lr * 0.5 * (1. + torch.cos(torch.pi * current_step / total_steps))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
 
    
# Training loop
total_steps = len(dataloader) * num_epochs
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
        
    for step, batch in enumerate(tqdm(dataloader)):
        
        optimizer.zero_grad()
        loss = model(batch)
        
        # Backpropagation
        loss.backward()
        # Learning Rate Scheduling
        current_step = epoch * len(dataloader) + step
        current_lr = lr_scheduler(optimizer, initial_lr, current_step, total_steps)
        
        optimizer.step()
        total_loss += loss.item()
        
    avg_loss = total_loss / len(dataloader)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')
        
print("Training Complete!")

# torch.cuda.empty_cache()
# torch.cuda.reset_max_memory_allocated()