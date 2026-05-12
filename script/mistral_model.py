import argparse
from transformers import MistralConfig, MistralForCausalLM
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
    untied_lm_head=True,
):
    head_dim = hidden_size // num_attention_heads
    k_dim = head_dim * num_key_value_heads
    fused_dim = hidden_size + 2 * k_dim

    embed = vocab_size * hidden_size
    per_layer = (
        2 * hidden_size
        + hidden_size * fused_dim
        + hidden_size * hidden_size
        + 3 * hidden_size * intermediate_size
    )
    final_norm = hidden_size
    lm_head = hidden_size * vocab_size if untied_lm_head else 0

    return embed + num_hidden_layers * per_layer + final_norm + lm_head


def build_config(exp):
    config = MistralConfig(
        vocab_size=8192,
        max_position_embeddings=4096,
        rope_theta=10000.0,
        sliding_window=4096,
        tie_word_embeddings=False,
    )

    apply_quantization = False
    simulation_seq_len = 16
    exp_name = ""

    if exp == 1:
        print("[SETUP] Exp 1: Routing Stress Test (GQA 12:3, High Traffic)")
        # Motivation:
        # Stress memory bandwidth and multicast behavior with GQA.
        # The 12:3 ratio preserves a 4x KV sharing factor.
        # Long sequence length creates sustained traffic pressure on the NoC.
        # 81,015,552 (~0.081 B) parameters
        exp_name = "Exp 1: Routing Stress Test (GQA 12:3, High Traffic)"
        config.hidden_size = 768
        config.intermediate_size = 3072
        config.num_hidden_layers = 8
        config.num_attention_heads = 12
        config.num_key_value_heads = 3
        simulation_seq_len = 256

    elif exp == 2:
        print("[SETUP] Exp 2: Compute Bound (Large Matrices, 1 Layer)")
        # Motivation:
        # Stress dense compute throughput with large GEMMs.
        # Symmetric attention removes routing asymmetry.
        # One layer minimizes pipeline/synchronization effects.
        # 80,745,728 (~0.081 B) parameters
        exp_name = "Exp 2: Compute Bound (Large Matrices, 1 Layer)"
        config.hidden_size = 1792
        config.intermediate_size = 7168
        config.num_hidden_layers = 1
        config.num_attention_heads = 16
        config.num_key_value_heads = 16
        simulation_seq_len = 16

    elif exp == 3:
        print("[SETUP] Exp 3: Pipeline & Sync Test (Small Matrices, 8 Layers)")
        # Motivation:
        # Small matrices make arithmetic cheap.
        # Multiple sequential layers expose synchronization overhead,
        # pipeline bubbles, and memory movement between stages.
        # 7,867,584 (~0.008 B) parameters
        exp_name = "Exp 3: Pipeline & Sync Test (Small Matrices, 8 Layers)"
        config.hidden_size = 192
        config.intermediate_size = 768
        config.num_hidden_layers = 8
        config.num_attention_heads = 6
        config.num_key_value_heads = 6
        simulation_seq_len = 16

    elif exp == 4:
        print("[SETUP] Exp 4: Full-Scale Simulation (~0.3 B, INT8)")
        # Motivation:
        # Larger realistic workload to stress memory footprint,
        # deeper execution paths, and numerical stability.
        # INT8 fake quantization is enabled for hardware-oriented evaluation.
        exp_name = "Exp 4: Full-Scale Simulation (~0.3 B, INT8)"

        config.hidden_size = 1024
        config.intermediate_size = 3072
        config.num_hidden_layers = 23
        config.num_attention_heads = 16
        config.num_key_value_heads = 8
        
        simulation_seq_len = 16
        apply_quantization = True

    else:
        raise ValueError(f"Unsupported experiment: {exp}")

    assert config.hidden_size % config.num_attention_heads == 0, \
        "hidden_size must be divisible by num_attention_heads"
    assert config.num_attention_heads % config.num_key_value_heads == 0, \
        "num_attention_heads must be divisible by num_key_value_heads"

    return config, simulation_seq_len, apply_quantization, exp_name


def main():
    parser = argparse.ArgumentParser(description="NoCDAS Experiment Generator (Mistral)")
    parser.add_argument(
        "--exp",
        type=int,
        choices=[1, 2, 3, 4],
        default=4,
        help="Choose the experiment: 1 (GQA Stress), 2 (Compute Bound), 3 (Pipeline), 4 (Full-Scale). Default: 4",
    )
    args = parser.parse_args()

    model_file = OUT_DIR / "mistral" / f"lm_transformer_mistral_exp{args.exp}.txt"
    weight_file = OUT_DIR / "mistral" / f"lm_weight_mistral_exp{args.exp}.txt"
    input_file = OUT_DIR / "mistral" / f"lm_input_mistral_exp{args.exp}.txt"

    os.makedirs(OUT_DIR / "mistral", exist_ok=True)

    config, SIMULATION_SEQ_LEN, apply_quantization, exp_name = build_config(args.exp)

    print(f"\n[INFO] Generating Mistral architecture for: {exp_name}")

    estimated_params = estimate_params(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        untied_lm_head=not config.tie_word_embeddings,
    )
    print(f"[INFO] Estimated parameters: {estimated_params:,} (~{estimated_params / 1e9:.3f} B)")

    model = MistralForCausalLM(config)
    model.eval()

    print(f"[INFO] Total generated parameters: {model.num_parameters():,}")

    VOCAB_SIZE = config.vocab_size
    D_MODEL = config.hidden_size
    NHEAD = config.num_attention_heads
    NUM_KV_HEADS = config.num_key_value_heads
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
    write_method(weight_file_open, model.model.embed_tokens.weight)

    # Transformer blocks
    for i, layer in enumerate(model.model.layers):
        topo_file.write(f"% --- Layer {i} ---\n")

        # RMSNorm 1
        write_node(f"RMSNorm {D_MODEL}")
        write_method(weight_file_open, layer.input_layernorm.weight)

        # Fused QKV projection
        k_dim = (D_MODEL // NHEAD) * NUM_KV_HEADS
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
        write_method(weight_file_open, layer.post_attention_layernorm.weight)

        # Fused MLP gate+up
        write_node(f"MatMul {D_MODEL} {DIM_FF * 2}")
        gate_up_weight = torch.cat([
            layer.mlp.gate_proj.weight,
            layer.mlp.up_proj.weight
        ], dim=0)
        write_method(weight_file_open, gate_up_weight, bias_tensor=None, needs_bias_slot=True)

        # SwiGLU
        write_node(f"SwiGLU {DIM_FF}")

        # Down projection
        write_node(f"MatMul {DIM_FF} {D_MODEL}")
        write_method(weight_file_open, layer.mlp.down_proj.weight, bias_tensor=None, needs_bias_slot=True)

        # Residual 2
        res_src = write_node(f"Add {D_MODEL} {res_src}")

        gc.collect()

    # Final output
    topo_file.write("% --- Final Output ---\n")
    write_node(f"RMSNorm {D_MODEL}")
    write_method(weight_file_open, model.model.norm.weight)

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