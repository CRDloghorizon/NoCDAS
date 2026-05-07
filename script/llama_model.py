from transformers import LlamaConfig, LlamaForCausalLM
from pathlib import Path
import numpy as np
import torch
import os
import gc

BASE_DIR    = Path(__file__).resolve().parents[1]
OUT_DIR     = BASE_DIR / "src" / "input"

model_file  = OUT_DIR / "lm_transformer_llama.txt"
weight_file = OUT_DIR / "lm_weight_llama.txt"
input_file  = OUT_DIR / "lm_input_llama.txt"

os.makedirs(OUT_DIR, exist_ok=True)

def write_weights_fp32(fw, tensor, bias_tensor=None, needs_bias_slot=False):
    """
    Standard high-precision FP32 export.
    Appends real biases at the end of each row, or 0.0 if the model is bias-less 
    but the hardware parser expects the extra column.
    """
    w_np = tensor.detach().cpu().float().numpy()
    b_np = bias_tensor.detach().cpu().float().numpy() if bias_tensor is not None else None
    
    if w_np.ndim == 1:
        # 1D layers like Normalizations
        fw.write(" ".join([f"{x:.6f}" for x in w_np]) + "\n")
    elif w_np.ndim == 2:
        # 2D layers like MatMul
        for i in range(w_np.shape[0]):
            row = w_np[i].tolist()
            if b_np is not None:
                row.append(b_np[i])
            elif needs_bias_slot:
                row.append(0.0)
            fw.write(" ".join([f"{x:.6f}" for x in row]) + "\n")

def write_weights_int8(fw, tensor, bias_tensor=None, needs_bias_slot=False):
    """
    Fake Quantization INT8 to simulate hardware error in large scale models.
    Biases are kept in FP32 to maintain numerical stability during inference.
    """
    w_np = tensor.detach().cpu().float().numpy()
    b_np = bias_tensor.detach().cpu().float().numpy() if bias_tensor is not None else None
    
    # 1D layers (Normalizations) remain in FP32
    if w_np.ndim == 1:
        fw.write(" ".join([f"{x:.6f}" for x in w_np]) + "\n")
        return

    # Scale calculation and INT8 quantization
    abs_max = np.max(np.abs(w_np))
    scale = float(abs_max / 127.0) if abs_max > 0 else 1.0 
    
    w_int8 = np.round(w_np / scale).astype(np.int8)
    w_dequant = (w_int8.astype(np.float32) * scale) 
    
    for i in range(w_dequant.shape[0]):
        row = w_dequant[i].tolist()
        if b_np is not None:
            row.append(b_np[i]) 
        elif needs_bias_slot:
            row.append(0.0)     
        fw.write(" ".join([f"{x:.6f}" for x in row]) + "\n")

def main():
    config = LlamaConfig(
        vocab_size=8192,
        hidden_size=1536,
        intermediate_size=4096,
        num_hidden_layers=11,
        num_attention_heads=16,
        num_key_value_heads=8,
        max_position_embeddings=4096,
        rope_theta=500000.0,
    )
    
    apply_quantization = True
    SIMULATION_SEQ_LEN = 64

    print("[INFO] Generating LLaMA model...")
    
    model = LlamaForCausalLM(config)
    model.eval()

    print(f"[INFO] Total number of generated parameters: {model.num_parameters():,}")
    
    VOCAB_SIZE = config.vocab_size
    D_MODEL = config.hidden_size
    NHEAD = config.num_attention_heads
    NUM_KV_HEADS = config.num_key_value_heads 
    DIM_FF = config.intermediate_size

    if apply_quantization is True:
        print(f"[INFO] Fake Quantization INT8.")
        write_method = write_weights_int8
    else:
        print(f"[INFO] Export in FP32.")
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

    # Input and embedding
    write_node(f"Input {SIMULATION_SEQ_LEN} 1 1")
    res_src = write_node(f"Embedding {VOCAB_SIZE} {D_MODEL}")
    write_method(weight_file_open, model.model.embed_tokens.weight)

    # Transformer blocks
    for i, layer in enumerate(model.model.layers): 
        topo_file.write(f"% --- Layer {i} ---\n") 

        # RMSNorm 1 (Pre-Attention)
        write_node(f"RMSNorm {D_MODEL}")
        write_method(weight_file_open, layer.input_layernorm.weight)

        # Fused MatMul for Attention (Q, K, V)
        k_dim = (D_MODEL // NHEAD) * NUM_KV_HEADS
        fused_dim = D_MODEL + 2 * k_dim
        write_node(f"MatMul {D_MODEL} {fused_dim}") 
        
        fused_weight = torch.cat([
            layer.self_attn.q_proj.weight, 
            layer.self_attn.k_proj.weight, 
            layer.self_attn.v_proj.weight
        ], dim=0)
        
        write_method(weight_file_open, fused_weight, bias_tensor=None, needs_bias_slot=True)
        
        # In-Transit Attention Node
        write_node(f"Attention {fused_dim} {D_MODEL} {k_dim} {NHEAD}")

        # Out Projection Attention (o_proj)
        write_node(f"MatMul {D_MODEL} {D_MODEL}")
        write_method(weight_file_open, layer.self_attn.o_proj.weight, bias_tensor=None, needs_bias_slot=True)

        # First Residual Connection
        res_src = write_node(f"Add {D_MODEL} {res_src}")

        # RMSNorm 2 (Post-Attention)
        write_node(f"RMSNorm {D_MODEL}")
        write_method(weight_file_open, layer.post_attention_layernorm.weight)

        # Fused MatMul for MLP (Gate + Up Projection)
        write_node(f"MatMul {D_MODEL} {DIM_FF * 2}")
        gate_up_weight = torch.cat([
            layer.mlp.gate_proj.weight, 
            layer.mlp.up_proj.weight
        ], dim=0)

        write_method(weight_file_open, gate_up_weight, bias_tensor=None, needs_bias_slot=True)
        
        # SwiGLU Node
        write_node(f"SwiGLU {DIM_FF}")
        
        # Down Projection MLP (down_proj)
        write_node(f"MatMul {DIM_FF} {D_MODEL}")
        write_method(weight_file_open, layer.mlp.down_proj.weight, bias_tensor=None, needs_bias_slot=True)

        # Second Residual Connection
        res_src = write_node(f"Add {D_MODEL} {res_src}")

        gc.collect()

    # Output
    topo_file.write(f"% --- Final Output ---\n")
    write_node(f"RMSNorm {D_MODEL}")
    write_method(weight_file_open, model.model.norm.weight)
    
    write_node(f"MatMul {D_MODEL} {VOCAB_SIZE}")
    write_method(weight_file_open, model.lm_head.weight, bias_tensor=None, needs_bias_slot=True)

    with open(input_file, "w") as fi:
        base_pattern = [10, 250, 314, 400, 50] 
        moltiplicatore = (SIMULATION_SEQ_LEN // len(base_pattern)) + 1
        test_sequence = (base_pattern * moltiplicatore)[:SIMULATION_SEQ_LEN]
        fi.write(" ".join([str(x) for x in test_sequence]) + "\n")

    topo_file.close()
    weight_file_open.close()
    
    print(f"\n[SUCCESS] Export completed successfully!")
    print(f"Topology saved to: {model_file}")
    print(f"Weights saved to: {weight_file}")
    print(f"Input saved to: {input_file}")

if __name__ == "__main__":
    main()