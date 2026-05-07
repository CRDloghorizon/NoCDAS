from models.LM_model import TransformerLM
from pathlib import Path
import numpy as np
import torch
import os
import gc

BASE_DIR    = Path(__file__).resolve().parents[1]
OUT_DIR     = BASE_DIR / "src" / "input"
path_file   = BASE_DIR / "script" / "models" / "LM.pth"

model_file  = OUT_DIR / "lm_transformer.txt"
weight_file = OUT_DIR / "lm_weight.txt"
input_file  = OUT_DIR / "lm_input.txt"

os.makedirs(OUT_DIR, exist_ok=True)

def write_weights_fp32(fw, tensor, bias_tensor=None, needs_bias_slot=False):
    w_np = tensor.detach().cpu().float().numpy()
    b_np = bias_tensor.detach().cpu().float().numpy() if bias_tensor is not None else None
    
    if w_np.ndim == 1:
        fw.write(" ".join([f"{x:.6f}" for x in w_np]) + "\n")
    elif w_np.ndim == 2:
        for i in range(w_np.shape[0]):
            row = w_np[i].tolist()
            if b_np is not None:
                row.append(b_np[i])
            elif needs_bias_slot:
                row.append(0.0)     # 0.0 if the layer is bias-less but C++ expects it
            fw.write(" ".join([f"{x:.6f}" for x in row]) + "\n")

def write_weights_int8(fw, tensor, bias_tensor=None, needs_bias_slot=False):
    """
    Fake Quantization INT8.
    WARNING: Biases do not get quantized to INT8, they remain in high precision (FP32) 
    to maintain numerical stability.
    """
    w_np = tensor.detach().cpu().float().numpy()
    b_np = bias_tensor.detach().cpu().float().numpy() if bias_tensor is not None else None
    
    # Normalization layer are in FP32
    if w_np.ndim == 1:
        fw.write(" ".join([f"{x:.6f}" for x in w_np]) + "\n")
        return

    abs_max = np.max(np.abs(w_np))
    scale = float(abs_max / 127.0) if abs_max > 0 else 1.0 
    
    # Quantization and de-quantization (INT8 simulation)
    w_int8 = np.round(w_np / scale).astype(np.int8)
    w_dequant = (w_int8.astype(np.float32) * scale) 
    
    for i in range(w_dequant.shape[0]):
        row = w_dequant[i].tolist()
        if b_np is not None:
            row.append(b_np[i]) # Real bias in FP32
        elif needs_bias_slot:
            row.append(0.0)     # Dummy bias
        fw.write(" ".join([f"{x:.6f}" for x in row]) + "\n")

def main():
    device = torch.device("cpu")
    
    if not os.path.exists(path_file):
        print(f"[ERROR] File '{path_file}' not found. Train the model first.")
        return

    print(f"[INFO] Reading checkpoint from '{path_file}'...")
    
    checkpoint = torch.load(path_file, map_location=device, weights_only=False)
    config = checkpoint['config']
    print(f"[INFO] Configuration from file: {config}")

    torch.set_default_dtype(torch.bfloat16)
    model = TransformerLM(**config).to(device)
    torch.set_default_dtype(torch.float32)
    
    model.load_state_dict(checkpoint['state_dict'])

    del checkpoint
    gc.collect()
    model.eval()

    SEQ_LEN = config['max_seq_len']
    VOCAB_SIZE = config['vocab_size']
    D_MODEL = config['d_model']
    NHEAD = config['nhead']
    NUM_KV_HEADS = config.get('num_kv_heads', NHEAD)
    DIM_FF = config['dim_feedforward']

    if D_MODEL >= 1024:
        print("\n--- [INFO] Big model detected. Applying Fake Quantization INT8 ---")
        write_method = write_weights_int8
    else:
        print("\n--- [INFO] Toy model detected. Applying standard FP32 export ---")
        write_method = write_weights_fp32

    topo_file = open(model_file, "w")
    weight_file_open = open(weight_file, "w")

    layer_counter = 0

    def write_node(line):
        nonlocal layer_counter
        topo_file.write(line + "\n")
        curr_id = layer_counter
        layer_counter += 1
        return curr_id

    # 1. INPUT E EMBEDDING
    write_node(f"Input {SEQ_LEN} 1 1")
    res_src = write_node(f"Embedding {VOCAB_SIZE} {D_MODEL}")
    write_method(weight_file_open, model.embedding.weight)

    # 2. TRANSFORMER BLOCKS
    for i, layer in enumerate(model.layers): 
        topo_file.write(f"% --- Layer {i} ---\n") 

        # RMSNorm 1
        write_node(f"RMSNorm {D_MODEL}")
        write_method(weight_file_open, layer.norm1.weight)

        # Fused Attention (Q, K, V) Weights
        k_dim = (D_MODEL // NHEAD) * NUM_KV_HEADS
        fused_dim = D_MODEL + 2 * k_dim
        write_node(f"MatMul {D_MODEL} {fused_dim}") 
        
        fused_weight = torch.cat([
            layer.attn.q_proj.weight, 
            layer.attn.k_proj.weight, 
            layer.attn.v_proj.weight
        ], dim=0)
        
        # Fused Attention (Q, K, V) Bias (if any)
        if layer.attn.q_proj.bias is not None:
            fused_bias = torch.cat([
                layer.attn.q_proj.bias, 
                layer.attn.k_proj.bias, 
                layer.attn.v_proj.bias
            ], dim=0)
        else:
            fused_bias = None

        write_method(weight_file_open, fused_weight, bias_tensor=fused_bias, needs_bias_slot=True)
        
        # Attention
        write_node(f"Attention {fused_dim} {D_MODEL} {k_dim} {NHEAD}")

        # Out Projection
        write_node(f"MatMul {D_MODEL} {D_MODEL}")
        write_method(weight_file_open, layer.attn.out_proj.weight, bias_tensor=layer.attn.out_proj.bias, needs_bias_slot=True)

        # Res Add 1
        res_src = write_node(f"Add {D_MODEL} {res_src}")

        # RMSNorm 2
        write_node(f"RMSNorm {D_MODEL}")
        write_method(weight_file_open, layer.norm2.weight)

        # SwiGLU Gate/Up
        write_node(f"MatMul {D_MODEL} {DIM_FF * 2}")
        write_method(weight_file_open, layer.gate_up_proj.weight, bias_tensor=layer.gate_up_proj.bias, needs_bias_slot=True)
        write_node(f"SwiGLU {DIM_FF}")
        
        # SwiGLU Down
        write_node(f"MatMul {DIM_FF} {D_MODEL}")
        write_method(weight_file_open, layer.down_proj.weight, bias_tensor=layer.down_proj.bias, needs_bias_slot=True)

        # Res Add 2
        res_src = write_node(f"Add {D_MODEL} {res_src}")

        gc.collect()

    # 3. OUTPUT
    topo_file.write(f"% --- Final Output ---\n")
    write_node(f"RMSNorm {D_MODEL}")
    write_method(weight_file_open, model.final_norm.weight)
    
    write_node(f"MatMul {D_MODEL} {VOCAB_SIZE}")
    write_method(weight_file_open, model.fc_out.weight, bias_tensor=model.fc_out.bias, needs_bias_slot=True)

    # 4. INPUT FILE
    with open(input_file, "w") as fi:
        base_pattern = [1, 2, 3, 4, 5]
        moltiplicatore = (SEQ_LEN // len(base_pattern)) + 1
        test_sequence = (base_pattern * moltiplicatore)[:SEQ_LEN]
        fi.write(" ".join([str(x) for x in test_sequence]) + "\n")

    topo_file.close()
    weight_file_open.close()
    
    print(f"[INFO] Export completed successfully!")

if __name__ == "__main__":
    main()