import torch
import torch.nn as nn
from Model import Transformer, Config
import tiktoken

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
enc = tiktoken.get_encoding('gpt2')

config = Config(
    n_vocab=enc.n_vocab,
    d_model=256,
    n_block=128,
    n_head=8,
    n_layer=12,
    d_inner=1024,
    dropout=0.2,
    bias=True
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Transformer(config).to(device)
model.load_state_dict(torch.load('model.pth'))

def generate_text(text, temperature=0.5):
    ctx = text
    ctx = enc.encode(ctx)
    ctx = torch.tensor(ctx, dtype=torch.long, device=device).unsqueeze(0)
    ctx = model.generate(ctx, 100, temperature)
    ctx = ctx.squeeze(0).tolist()
    ctx = enc.decode(ctx)
    return ctx

prompt = ["Scarlett looked up quickly",
"She smiled, with",
"With a ",
"She want's to, but",
"You're stupid!",
"No, they'll",
"I forgot about that",
"Since the return of ",
"It was not that",
"She, too"]

for p in prompt:
    print(f"PROMPT: {p}")
    print(f"Generated: {generate_text(p)}")
