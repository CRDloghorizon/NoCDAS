import math
import argparse
from transformers import GemmaConfig, GemmaForCausalLM
from pathlib import Path
import numpy as np
import torch
import os
import gc

BASE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = BASE_DIR / "src" / "input"


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
    w_dequant = w_int8.astype(np.float32) * scale

    for i in range(w_dequant.shape[0]):
        row = w_dequant[i].tolist()
        if b_np is not None:
            row.append(b_np[i])
        elif needs_bias_slot:
            row.append(0.0)
        fw.write(" ".join([f"{x:.6f}" for x in row]) + "\n")


def estimate_params(
    vocab_size,
    hidden_size,
    intermediate_size,
    num_hidden_layers,
    num_attention_heads,
    num_key_value_heads,
    head_dim,
    tied_lm_head=True,
):
    q_dim = head_dim * num_attention_heads
    k_dim = head_dim * num_key_value_heads
    fused_dim = q_dim + 2 * k_dim

    embed = vocab_size * hidden_size
    per_layer = (
        2 * hidden_size
        + hidden_size * fused_dim
        + q_dim * hidden_size
        + 3 * hidden_size * intermediate_size
    )
    final_norm = hidden_size
    lm_head = 0 if tied_lm_head else hidden_size * vocab_size

    return embed + num_hidden_layers * per_layer + final_norm + lm_head


def build_config(exp):
    config = GemmaConfig(
        vocab_size=8192,
        max_position_embeddings=4096,
        rms_norm_eps=1e-6,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        tie_word_embeddings=True,
        attention_bias=False,
        hidden_act="gelu_pytorch_tanh",
    )

    apply_quantization = False
    simulation_seq_len = 16
    exp_name = ""

    if exp == 1:
        print("[SETUP] Exp 1: Routing Stress Test (GQA 14:2, High Traffic)")
        # Motivation:
        # Preserve GQA behavior with strong KV sharing and long sequence length.
        # This setup emphasizes memory traffic and multicast pressure on the NoC.
        # 99,105,664 (~0.099 B) parameters
        exp_name = "Exp 1: Routing Stress Test (GQA 14:2, High Traffic)"
        config.hidden_size = 896
        config.intermediate_size = 3584
        config.num_hidden_layers = 8
        config.num_attention_heads = 14
        config.num_key_value_heads = 2
        config.head_dim = 64
        simulation_seq_len = 256

    elif exp == 2:
        print("[SETUP] Exp 2: Compute Bound (Large Matrices, 1 Layer)")
        # Motivation:
        # Symmetric attention and one layer minimize routing and pipeline effects.
        # Large GEMMs dominate execution time and stress raw compute throughput.
        # 83,892,224 (~0.084 B) parameters
        exp_name = "Exp 2: Compute Bound (Large Matrices, 1 Layer)"
        config.hidden_size = 2048
        config.intermediate_size = 8192
        config.num_hidden_layers = 1
        config.num_attention_heads = 16
        config.num_key_value_heads = 16
        config.head_dim = 128
        simulation_seq_len = 16

    elif exp == 3:
        print("[SETUP] Exp 3: Pipeline & Sync Test (Small Matrices, 8 Layers)")
        # Motivation:
        # Small matrices make arithmetic cheap.
        # Multiple sequential layers emphasize synchronization,
        # pipeline bubbles, and inter-layer movement overhead.
        # 6,294,720 (~0.006 B) parameters
        exp_name = "Exp 3: Pipeline & Sync Test (Small Matrices, 8 Layers)"
        config.hidden_size = 192
        config.intermediate_size = 768
        config.num_hidden_layers = 8
        config.num_attention_heads = 6
        config.num_key_value_heads = 6
        config.head_dim = 32
        simulation_seq_len = 16

    elif exp == 4:
        print("[SETUP] Exp 4: Full-Scale Simulation (~0.28 B, INT8)")
        # Motivation:
        # Larger realistic workload for memory footprint,
        # longer execution paths, and numerical robustness.
        # INT8 fake quantization approximates large-scale hardware deployment effects.
        # 282,104,832 (~0.282 B) parameters
        exp_name = "Exp 4: Full-Scale Simulation (~0.28 B, INT8)"
        config.hidden_size = 1024
        config.intermediate_size = 4096
        config.num_hidden_layers = 18
        config.num_attention_heads = 16
        config.num_key_value_heads = 4
        config.head_dim = 64
        simulation_seq_len = 64
        apply_quantization = True

    else:
        raise ValueError(f"Unsupported experiment: {exp}")

    assert config.hidden_size % config.num_attention_heads == 0, \
        "hidden_size must be divisible by num_attention_heads"
    assert config.num_attention_heads % config.num_key_value_heads == 0, \
        "num_attention_heads must be divisible by num_key_value_heads"
    assert config.head_dim * config.num_attention_heads == config.hidden_size, \
        "For this exporter, require head_dim * num_attention_heads == hidden_size"

    return config, simulation_seq_len, apply_quantization, exp_name


def main():
    parser = argparse.ArgumentParser(description="NoCDAS Experiment Generator (Gemma)")
    parser.add_argument(
        "--exp",
        type=int,
        choices=[1, 2, 3, 4],
        default=4,
        help="Choose the experiment: 1 (GQA Stress), 2 (Compute Bound), 3 (Pipeline), 4 (Full-Scale). Default: 4",
    )
    args = parser.parse_args()

    model_file = OUT_DIR / "gemma" / f"lm_transformer_gemma_exp{args.exp}.txt"
    weight_file = OUT_DIR / "gemma" / f"lm_weight_gemma_exp{args.exp}.txt"
    input_file = OUT_DIR / "gemma" / f"lm_input_gemma_exp{args.exp}.txt"

    os.makedirs(OUT_DIR / "gemma", exist_ok=True)

    config, SIMULATION_SEQ_LEN, apply_quantization, exp_name = build_config(args.exp)

    print(f"\n[INFO] Generating Gemma architecture for: {exp_name}")

    estimated_params = estimate_params(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        tied_lm_head=config.tie_word_embeddings,
    )
    print(f"[INFO] Estimated parameters: {estimated_params:,} (~{estimated_params / 1e9:.3f} B)")

    model = GemmaForCausalLM(config)
    model.eval()

    print(f"[INFO] Total generated parameters: {model.num_parameters():,}")

    VOCAB_SIZE = config.vocab_size
    D_MODEL = config.hidden_size
    NHEAD = config.num_attention_heads
    NUM_KV_HEADS = config.num_key_value_heads
    HEAD_DIM = config.head_dim
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

    # Input + embedding
    write_node(f"Input {SIMULATION_SEQ_LEN} 1 1")
    res_src = write_node(f"Embedding {VOCAB_SIZE} {D_MODEL}")
    # write_method(weight_file_open, model.model.embed_tokens.weight)
    embed_scaled = model.model.embed_tokens.weight * math.sqrt(D_MODEL)
    write_method(weight_file_open, embed_scaled)

    # Transformer blocks
    for i, layer in enumerate(model.model.layers):
        topo_file.write(f"% --- Layer {i} ---\n")

        # RMSNorm 1
        write_node(f"RMSNorm {D_MODEL}")
        write_method(weight_file_open, layer.input_layernorm.weight+1.0)

        # Fused QKV projection
        k_dim = HEAD_DIM * NUM_KV_HEADS
        fused_dim = D_MODEL + 2 * k_dim
        write_node(f"MatMul {D_MODEL} {fused_dim}")

        fused_weight = torch.cat([
            layer.self_attn.q_proj.weight,
            layer.self_attn.k_proj.weight,
            layer.self_attn.v_proj.weight
        ], dim=0)

        write_method(weight_file_open, fused_weight, bias_tensor=None, needs_bias_slot=True)

        # Attention
        write_node(f"Attention {fused_dim} {D_MODEL} {k_dim} {NHEAD}")

        # Output projection
        write_node(f"MatMul {D_MODEL} {D_MODEL}")
        write_method(weight_file_open, layer.self_attn.o_proj.weight, bias_tensor=None, needs_bias_slot=True)

        # Residual 1
        res_src = write_node(f"Add {D_MODEL} {res_src}")

        # RMSNorm 2
        write_node(f"RMSNorm {D_MODEL}")
        write_method(weight_file_open, layer.post_attention_layernorm.weight + 1.0)

        # Fused MLP gate+up
        write_node(f"MatMul {D_MODEL} {DIM_FF * 2}")
        gate_up_weight = torch.cat([
            layer.mlp.gate_proj.weight,
            layer.mlp.up_proj.weight
        ], dim=0)

        write_method(weight_file_open, gate_up_weight, bias_tensor=None, needs_bias_slot=True)

        # GeGLU
        write_node(f"GeGLU {DIM_FF}")

        # Down projection
        write_node(f"MatMul {DIM_FF} {D_MODEL}")
        write_method(weight_file_open, layer.mlp.down_proj.weight, bias_tensor=None, needs_bias_slot=True)

        # Residual 2
        res_src = write_node(f"Add {D_MODEL} {res_src}")

        gc.collect()

    # Final output
    topo_file.write("% --- Final Output ---\n")
    write_node(f"RMSNorm {D_MODEL}")
    write_method(weight_file_open, model.model.norm.weight + 1.0)

    write_node(f"MatMul {D_MODEL} {VOCAB_SIZE}")
    write_method(weight_file_open, model.lm_head.weight, bias_tensor=None, needs_bias_slot=True)

    with open(input_file, "w") as fi:
        base_pattern = [10, 250, 314, 400, 50, 77, 88, 12, 1024, 8000]
        multiplier = (SIMULATION_SEQ_LEN // len(base_pattern)) + 1
        test_sequence = (base_pattern * multiplier)[:SIMULATION_SEQ_LEN]
        fi.write(" ".join([str(x) for x in test_sequence]) + "\n")

    topo_file.close()
    weight_file_open.close()

    print(f"\n[SUCCESS] Export completed successfully!")
    print(f"Topology saved to: {model_file}")
    print(f"Weights saved to: {weight_file}")
    print(f"Input saved to: {input_file}")


if __name__ == "__main__":
    main()
