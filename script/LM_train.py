from models.LM_model import TransformerLM
import torch.optim as optim
from pathlib import Path
import torch.nn as nn
import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {device.type.upper()}")

# # 3.8B parameters.
# VOCAB_SIZE   = 32000
# D_MODEL      = 3072
# NHEAD        = 24
# NUM_KV_HEADS = 8
# NUM_LAYERS   = 32
# DIM_FF       = 8192
# SEQ_LEN      = 2048
# BATCH_SIZE   = 1
# EPOCHS       = 100
# lr           = 0.0005
# path_file = Path(__file__).parent.resolve() / "models" / "LM_3B.pth"
# torch.set_default_dtype(torch.bfloat16)

# # Toy Model -> 127.000 parameters.
# VOCAB_SIZE   = 20
# D_MODEL      = 64
# NHEAD        = 2
# NUM_KV_HEADS = 1   
# NUM_LAYERS   = 2
# DIM_FF       = 256
# SEQ_LEN      = 15
# BATCH_SIZE   = 8
# EPOCHS       = 100
# lr           = 0.005
# path_file = Path(__file__).parent.resolve() / "models" / "LM.pth"

# 1M parameters.
VOCAB_SIZE   = 1024
D_MODEL      = 128
NHEAD        = 4
NUM_KV_HEADS = 2
NUM_LAYERS   = 4
DIM_FF       = 384
SEQ_LEN      = 128
BATCH_SIZE   = 8
EPOCHS       = 100
lr           = 0.001
path_file = Path(__file__).parent.resolve() / "models" / "LM_1M.pth"

model = TransformerLM(
    vocab_size      = VOCAB_SIZE, 
    d_model         = D_MODEL, 
    nhead           = NHEAD, 
    num_kv_heads    = NUM_KV_HEADS,
    num_layers      = NUM_LAYERS,
    dim_feedforward = DIM_FF,
    max_seq_len     = SEQ_LEN
).to(device)

# Cast to float32 (in case it was changed for the 3.8B model)
torch.set_default_dtype(torch.float32)

if os.path.exists(path_file):
    print(f"\n[INFO] Found the checkpoint: '{path_file}'.")
    checkpoint = torch.load(path_file, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint['state_dict'])

else:
    print(f"\n[INFO] File '{path_file}' NOT found, starting training.")
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Dummy data for training.
    pattern = [1, 2, 3, 4, 5]
    full_data = torch.tensor(pattern * 1000, dtype=torch.long)

    model.train()
    for epoca in range(EPOCHS):
        indici_start = torch.randint(0, len(full_data) - SEQ_LEN - 1, (BATCH_SIZE,))
        
        x = torch.stack([full_data[i : i + SEQ_LEN] for i in indici_start]).to(device)
        y = torch.stack([full_data[i + 1 : i + SEQ_LEN + 1] for i in indici_start]).to(device)
        
        optimizer.zero_grad()
        
        # Mixed Precision
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            output = model(x)
            output_flat = output.view(-1, VOCAB_SIZE)
            y_flat = y.view(-1)
            loss = criterion(output_flat, y_flat)
        
        loss.backward()
        optimizer.step()
        
        if (epoca + 1) % 5 == 0:
            print(f"Epoch [{epoca+1}/{EPOCHS}], Loss: {loss.item():.4f}")

    checkpoint = {
        'config': {
            'vocab_size': VOCAB_SIZE,
            'd_model': D_MODEL,
            'nhead': NHEAD,
            'num_kv_heads': NUM_KV_HEADS,
            'num_layers': NUM_LAYERS,
            'dim_feedforward': DIM_FF,
            'max_seq_len': SEQ_LEN
        },
        'state_dict': model.state_dict()
    }
    
    torch.save(checkpoint, path_file)
    print(f"\n[INFO] Checkpoint (config + weights) saved in: {path_file}")

print("\n--- INFERENCE ---")
model.eval()

with torch.no_grad(): 
    prompt = torch.tensor([[1]], dtype=torch.long).to(device)
    gen_len = 10
    
    print(f"Starting prompt: {prompt[0].tolist()}")
    
    for _ in range(gen_len):
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            prediction = model(prompt)
            last_token_logits = prediction[:, -1, :] 
            
        new_token = torch.argmax(last_token_logits, dim=-1, keepdim=True)
        prompt = torch.cat([prompt, new_token], dim=1)

print(f"Output sequence: {prompt[0].tolist()}")