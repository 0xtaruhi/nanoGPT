import tiktoken
from Model import Transformer, Config
import torch
import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
enc = tiktoken.get_encoding('gpt2')

# Define Transformer model parameters
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

# Init train data
with open('gone_with_the_wind.txt', 'r') as f:
    text = f.read()
text = enc.encode(text)

data = torch.tensor(text, dtype=torch.long, device=device)
n = int(0.8 * len(data))
train_data, val_data = data[:n], data[n:]

batch_size = 128
def get_batch(split):
    data = train_data if split == 'train' else val_data
    
    ix = torch.randint(0, len(data) - config.n_block, (batch_size,))
    x = torch.stack([data[i:i+config.n_block] for i in ix])
    y = torch.stack([data[i+1:i+config.n_block+1] for i in ix])
    x, y = x.to(device), y.to(device)
    
    return x, y

# Model initialization
model = Transformer(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

def generate_text(text):
    idx = torch.tensor([enc.encode(text)], dtype=torch.long).to(device)
    generated_tokens = model.generate(idx, 100, 0.5)
    generated_text = enc.decode(generated_tokens.tolist()[0])
    return generated_text

for epoch in tqdm.tqdm(range(1500)):
    x, y = get_batch('train')
    logits, loss = model(x, y)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
        
    if (epoch + 1) % 300 == 0:
        times = 10
        total_loss = 10
        for _ in range(times):
            x, y = get_batch('val')
            logits, loss = model(x, y)
            total_loss += loss.item()
        print(f'Validation loss: {total_loss / times:.4f}')

torch.save(model.state_dict(), 'model.pth')
