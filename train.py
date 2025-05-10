# train.py
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm.notebook import tqdm

# --- طبقات النموذج ---
class OPTAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        attn_scores = (q @ k.transpose(-2, -1)) / self.scale
        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        out = attn_probs @ v
        out = out.transpose(1,2).contiguous().view(B, T, C)
        return self.out_proj(out)

class OPTDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = OPTAttention(embed_dim, num_heads, dropout)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.ln2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        res = x
        x = self.attn(x)
        x = self.dropout(x)
        x = self.ln1(x + res)
        res = x
        x = self.ff(x)
        x = self.dropout(x)
        x = self.ln2(x + res)
        return x

class OPTModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=768, num_heads=12, num_layers=6, ff_dim=3072, max_len=128, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            OPTDecoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, input_ids):
        B, T = input_ids.size()
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        return x

class OPTForCausalLM(nn.Module):
    def __init__(self, vocab_size, **kwargs):
        super().__init__()
        self.opt = OPTModel(vocab_size, **kwargs)
        self.lm_head = nn.Linear(kwargs.get('embed_dim',768), vocab_size, bias=False)

    def forward(self, input_ids, labels=None):
        hidden = self.opt(input_ids)
        logits = self.lm_head(hidden)
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return loss, logits
        return logits

# --- التدريب ---
def preprocess_batch(batch, tokenizer, max_length=128):
    texts = [" [SEP] ".join(d) if isinstance(d, list) else d for d in batch["dialog"]]
    tokens = tokenizer(texts, padding='max_length', truncation=True, max_length=max_length)
    return {'input_ids': tokens['input_ids'], 'labels': tokens['input_ids']}

class ChatDataset(torch.utils.data.Dataset):
    def __init__(self, dataset): self.dataset = dataset
    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx):
        item = self.dataset[idx]
        return torch.tensor(item['input_ids']), torch.tensor(item['labels'])

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv02")
    raw_datasets = load_dataset("daily_dialog")

    encoded_train = raw_datasets["train"].map(lambda b: preprocess_batch(b, tokenizer), batched=True, remove_columns=raw_datasets["train"].column_names)
    encoded_test = raw_datasets["test"].map(lambda b: preprocess_batch(b, tokenizer), batched=True, remove_columns=raw_datasets["test"].column_names)

    train_loader = DataLoader(ChatDataset(encoded_train), batch_size=8, shuffle=True)
    model = OPTForCausalLM(
        vocab_size=tokenizer.vocab_size,
        embed_dim=768,
        num_heads=12,
        num_layers=6,
        ff_dim=3072,
        max_len=128
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    for epoch in range(3):
        model.train()
        total_loss = 0
        for input_ids, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            input_ids, labels = input_ids.to(device), labels.to(device)
            optimizer.zero_grad()
            loss, _ = model(input_ids, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} - Loss: {total_loss / len(train_loader):.4f}")

    model_dir = "/kaggle/working/opt_model"
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, "pytorch_model.bin"))
    tokenizer.save_pretrained(model_dir)
