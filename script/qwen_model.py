from transformers import AutoModelForCausalLM, AutoConfig
from pathlib import Path
import numpy as np
import torch
import os
import gc

BASE_DIR    = Path(__file__).resolve().parents[1]
OUT_DIR     = BASE_DIR / "src" / "input"

model_file  = OUT_DIR / "lm_transformer.txt"
weight_file = OUT_DIR / "lm_weight.txt"
input_file  = OUT_DIR / "lm_input.txt"

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
    device = torch.device("cpu")
    
    config = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B")
    
    config.vocab_size = 8192
    SIMULATION_SEQ_LEN = 64

    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32)
    model.eval()

    print(f"[INFO] Number of parameters: {model.num_parameters():,}")
    
    VOCAB_SIZE = config.vocab_size
    D_MODEL = config.hidden_size
    NHEAD = config.num_attention_heads
    NUM_KV_HEADS = getattr(config, 'num_key_value_heads', NHEAD) 
    DIM_FF = config.intermediate_size

    # Auto-select export method based on model size
    if D_MODEL >= 896:
        print(f"[INFO] Full-scale model detected (D_MODEL={D_MODEL}). Applying INT8 Quantization.")
        write_method = write_weights_int8
    else:
        print(f"[INFO] Compact model detected (D_MODEL={D_MODEL}). Applying pure FP32 export.")
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

    print("\n[INFO] Starting topological and weight files generation...")

    # Input and embedding
    write_node(f"Input {SIMULATION_SEQ_LEN} 1 1")
    res_src = write_node(f"Embedding {VOCAB_SIZE} {D_MODEL}")
    write_method(weight_file_open, model.model.embed_tokens.weight)

    # Loop over transformer blocks (Qwen layers)
    for i, layer in enumerate(model.model.layers): 
        topo_file.write(f"% --- Layer {i} ---\n") 

        # RMSNorm 1 (Pre-Attention)
        write_node(f"RMSNorm {D_MODEL}")
        write_method(weight_file_open, layer.input_layernorm.weight)

        # Fused MatMul for Attention (Q, K, V)
        k_dim = (D_MODEL // NHEAD) * NUM_KV_HEADS
        fused_dim = D_MODEL + 2 * k_dim
        write_node(f"MatMul {D_MODEL} {fused_dim}") 
        
        # Extract and concatenate Q, K, V Weights
        fused_weight = torch.cat([
            layer.self_attn.q_proj.weight, 
            layer.self_attn.k_proj.weight, 
            layer.self_attn.v_proj.weight
        ], dim=0)
        
        # Extract and concatenate Q, K, V Biases (if used by the model)
        if hasattr(layer.self_attn.q_proj, 'bias') and layer.self_attn.q_proj.bias is not None:
            fused_bias = torch.cat([
                layer.self_attn.q_proj.bias, 
                layer.self_attn.k_proj.bias, 
                layer.self_attn.v_proj.bias
            ], dim=0)
        else:
            fused_bias = None

        write_method(weight_file_open, fused_weight, bias_tensor=fused_bias, needs_bias_slot=True)
        
        # In-Transit Attention Node
        write_node(f"Attention {fused_dim} {D_MODEL} {k_dim} {NHEAD}")

        # Out Projection Attention (o_proj)
        write_node(f"MatMul {D_MODEL} {D_MODEL}")
        out_bias = layer.self_attn.o_proj.bias if hasattr(layer.self_attn.o_proj, 'bias') else None
        write_method(weight_file_open, layer.self_attn.o_proj.weight, bias_tensor=out_bias, needs_bias_slot=True)

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
        
        if hasattr(layer.mlp.gate_proj, 'bias') and layer.mlp.gate_proj.bias is not None:
            gate_up_bias = torch.cat([layer.mlp.gate_proj.bias, layer.mlp.up_proj.bias], dim=0)
        else:
            gate_up_bias = None

        write_method(weight_file_open, gate_up_weight, bias_tensor=gate_up_bias, needs_bias_slot=True)
        
        # SwiGLU Node
        write_node(f"SwiGLU {DIM_FF}")
        
        # Down Projection MLP (down_proj)
        write_node(f"MatMul {DIM_FF} {D_MODEL}")
        down_bias = layer.mlp.down_proj.bias if hasattr(layer.mlp.down_proj, 'bias') else None
        write_method(weight_file_open, layer.mlp.down_proj.weight, bias_tensor=down_bias, needs_bias_slot=True)

        # Second Residual Connection
        res_src = write_node(f"Add {D_MODEL} {res_src}")

        gc.collect()

    # Final output (LayerNorm and LM Head)
    topo_file.write(f"% --- Final Output ---\n")
    write_node(f"RMSNorm {D_MODEL}")
    write_method(weight_file_open, model.model.norm.weight)
    
    write_node(f"MatMul {D_MODEL} {VOCAB_SIZE}")
    lm_head_bias = model.lm_head.bias if hasattr(model.lm_head, 'bias') else None
    write_method(weight_file_open, model.lm_head.weight, bias_tensor=lm_head_bias, needs_bias_slot=True)

    # Simulated input file generation
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