import argparse
from transformers import AutoModelForCausalLM, AutoConfig
from pathlib import Path
import numpy as np
import torch
import os
import gc

BASE_DIR    = Path(__file__).resolve().parents[1]
OUT_DIR     = BASE_DIR / "src" / "input"

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
                row.append(0.0)
            fw.write(" ".join([f"{x:.6f}" for x in row]) + "\n")

def write_weights_int8(fw, tensor, bias_tensor=None, needs_bias_slot=False):
    w_np = tensor.detach().cpu().float().numpy()
    b_np = bias_tensor.detach().cpu().float().numpy() if bias_tensor is not None else None
    
    if w_np.ndim == 1:
        fw.write(" ".join([f"{x:.6f}" for x in w_np]) + "\n")
        return

    abs_max = np.max(np.abs(w_np), axis=1, keepdims=True)
    scale = np.where(abs_max > 0, abs_max / 127.0, 1.0)
    
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
    parser = argparse.ArgumentParser(description="NoCDAS Experiment Generator (Qwen)")
    parser.add_argument("--exp", type=int, choices=[1, 2, 3, 4], default=4,
                        help="Choose the experiment: 1 (GQA Stress), 2 (Compute Bound), 3 (Pipeline), 4 (Full-Scale). Default: 4")
    args = parser.parse_args()

    model_file  = OUT_DIR / "qwen" / f"lm_transformer_qwen_exp{args.exp}.txt"
    weight_file = OUT_DIR / "qwen" / f"lm_weight_qwen_exp{args.exp}.txt"
    input_file  = OUT_DIR / "qwen" / f"lm_input_qwen_exp{args.exp}.txt"

    os.makedirs(OUT_DIR / "qwen", exist_ok=True)

    config = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B")
    apply_quantization = True

    config.vocab_size = 8192

    if args.exp == 1:
        print("[SETUP] Exp 1: Routing Stress Test (GQA 8:2, High Traffic)")
            # Experiment 1: routing stress test (GQA and Data Traffic)
            # # Motivation (Memory Bandwidth and Multicast): This experiment tests 
            # Grouped-Query Attention (GQA). Having 8 heads for Queries but only 2 
            # for Key and Value, your hardware will have to read a single KV block 
            # from memory and "spread" it (broadcasting/multicasting) across 4 different 
            # Query compute units. This is used to see if the network-on-chip (NoC) 
            # or the data bus gets congested when performing this asymmetric routing. 
            # The sequence length of 256 is needed to generate enough token traffic to 
            # stress the bandwidth.
            # 13,077,888 parameters -> 0.013 B
        config.num_hidden_layers = 1
        config.hidden_size = 768
        config.intermediate_size = 2304
        config.num_attention_heads = 8
        config.num_key_value_heads = 2
        SIMULATION_SEQ_LEN = 256

    elif args.exp == 2:
        print("[SETUP] Exp 2: Compute Bound (Large Matrices, 2 Layer)")
            # # Experiment 2: compute bound
            # # Motivation (ALU Saturation): This setup eliminates routing issues (the 
            # heads are symmetric, 8 to 8) and latency (there is only one layer). The 
            # goal is to create massive matrices to keep the hardware's multiply-and-accumulate 
            # (MAC) units 100% occupied. It measures the raw peak TFLOPS of your accelerator. 
            # It is "compute bound" because the execution time will be limited purely by how 
            # fast the chip can perform multiplications, not by how fast it can move data.
            # 83,898,368 parameyets -> 0.084 B
        config.num_hidden_layers = 1
        config.hidden_size = 1024
        config.intermediate_size = 4096
        config.num_attention_heads = 16
        config.num_key_value_heads = 16
        SIMULATION_SEQ_LEN = 16

    elif args.exp == 3:
        print("[SETUP] Exp 3: Pipeline & Sync Test (Small Matrices, 8 Layer)")
            # # Experiment 3: Pipeline test
            # # Motivation (Latency and Synchronization): Exactly the opposite of Experiment 2. 
            # The matrices are very small, so the computations finish in an instant. However, 
            # since there are 8 layers in sequence, the hardware must constantly stop, write 
            # partial results, synchronize, and move to the next layer. This experiment reveals 
            # the latency of the chip, the efficiency of its pipeline, and how much time it "wastes" 
            # in overhead between one operation and another (e.g., read/write accesses to internal 
            # cache/SRAM memories).
            # 18,895,104 parameters -> 0.019 B
        config.num_hidden_layers = 16
        config.hidden_size = 256
        config.intermediate_size = 1024
        config.num_attention_heads = 8
        config.num_key_value_heads = 8
        SIMULATION_SEQ_LEN = 16

    elif args.exp == 4:
        print("[SETUP] Exp 4: Full-Scale Simulation (~0.36 B)")
            # # Experiment 4: Full-Scale Simulation (INT8 Quantization)
            # # Motivation (Realistic Workload and Memory): This is the "real" test. It combines 
            # considerable depth and significant width to simulate the total memory footprint 
            # (DRAM or HBM) and test the numerical stability of the system over long computation paths.
            # Total parameters = 365,238,144 -> 0.36 B
        apply_quantization = True
        SIMULATION_SEQ_LEN = 64

    print(f"\n[INFO] Generating Qwen architecture...")
    model = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
    model.eval()

    print(f"[INFO] Total parameters: {model.num_parameters():,}")
    
    VOCAB_SIZE = config.vocab_size
    D_MODEL = config.hidden_size
    NHEAD = config.num_attention_heads
    NUM_KV_HEADS = getattr(config, 'num_key_value_heads', NHEAD) 
    DIM_FF = config.intermediate_size

    write_method = write_weights_int8 if apply_quantization else write_weights_fp32
    print(f"[INFO] INT8 quantization (Per-Row) on: {apply_quantization}")

    topo_file = open(model_file, "w")
    weight_file_open = open(weight_file, "w") 
    layer_counter = 0

    def write_node(line):
        nonlocal layer_counter
        topo_file.write(line + "\n")
        curr_id = layer_counter
        layer_counter += 1
        return curr_id

    write_node(f"Input {SIMULATION_SEQ_LEN} 1 1")
    res_src = write_node(f"Embedding {VOCAB_SIZE} {D_MODEL}")
    write_method(weight_file_open, model.model.embed_tokens.weight)

    for i, layer in enumerate(model.model.layers): 
        topo_file.write(f"% --- Layer {i} ---\n") 

        write_node(f"RMSNorm {D_MODEL}")
        write_method(weight_file_open, layer.input_layernorm.weight)

        k_dim = (D_MODEL // NHEAD) * NUM_KV_HEADS
        fused_dim = D_MODEL + 2 * k_dim
        write_node(f"MatMul {D_MODEL} {fused_dim}") 
        
        fused_weight = torch.cat([layer.self_attn.q_proj.weight, layer.self_attn.k_proj.weight, layer.self_attn.v_proj.weight], dim=0)
        
        if hasattr(layer.self_attn.q_proj, 'bias') and layer.self_attn.q_proj.bias is not None:
            fused_bias = torch.cat([layer.self_attn.q_proj.bias, layer.self_attn.k_proj.bias, layer.self_attn.v_proj.bias], dim=0)
        else:
            fused_bias = None

        write_method(weight_file_open, fused_weight, bias_tensor=fused_bias, needs_bias_slot=True)
        write_node(f"Attention {fused_dim} {D_MODEL} {k_dim} {NHEAD}")

        write_node(f"MatMul {D_MODEL} {D_MODEL}")
        out_bias = layer.self_attn.o_proj.bias if hasattr(layer.self_attn.o_proj, 'bias') else None
        write_method(weight_file_open, layer.self_attn.o_proj.weight, bias_tensor=out_bias, needs_bias_slot=True)

        res_src = write_node(f"Add {D_MODEL} {res_src}")

        write_node(f"RMSNorm {D_MODEL}")
        write_method(weight_file_open, layer.post_attention_layernorm.weight)

        write_node(f"MatMul {D_MODEL} {DIM_FF * 2}")
        gate_up_weight = torch.cat([layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight], dim=0)
        
        if hasattr(layer.mlp.gate_proj, 'bias') and layer.mlp.gate_proj.bias is not None:
            gate_up_bias = torch.cat([layer.mlp.gate_proj.bias, layer.mlp.up_proj.bias], dim=0)
        else:
            gate_up_bias = None

        write_method(weight_file_open, gate_up_weight, bias_tensor=gate_up_bias, needs_bias_slot=True)
        write_node(f"SwiGLU {DIM_FF}")
        
        write_node(f"MatMul {DIM_FF} {D_MODEL}")
        down_bias = layer.mlp.down_proj.bias if hasattr(layer.mlp.down_proj, 'bias') else None
        write_method(weight_file_open, layer.mlp.down_proj.weight, bias_tensor=down_bias, needs_bias_slot=True)

        res_src = write_node(f"Add {D_MODEL} {res_src}")
        gc.collect()

    topo_file.write(f"% --- Final Output ---\n")
    write_node(f"RMSNorm {D_MODEL}")
    write_method(weight_file_open, model.model.norm.weight)
    
    write_node(f"MatMul {D_MODEL} {VOCAB_SIZE}")
    lm_head_bias = model.lm_head.bias if hasattr(model.lm_head, 'bias') else None
    write_method(weight_file_open, model.lm_head.weight, bias_tensor=lm_head_bias, needs_bias_slot=True)

    with open(input_file, "w") as fi:
        base_pattern = [10, 250, 314, 400, 50, 77, 88, 12, 1024, 8000] 
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