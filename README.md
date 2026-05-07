# NoCDAS
**A Cycle-Accurate NoC-based Deep Neural Network Accelerator Simulator**

NoCDAS is a cycle-accurate simulation framework designed to evaluate the performance, latency, and power characteristics of hardware accelerators for Deep Neural Networks (DNNs). 

Originally focused on Convolutional Neural Networks (CNNs) using a standard *Compute-at-PE* architecture, NoCDAS has been extensively upgraded to support Large Language Model (LLM) inference. While LLMs can theoretically run on traditional architectures, NoCDAS introduces the computational Network-on-Chip (cNoC) paradigm as a powerful hardware optimization. By enabling in-transit computation, the cNoC approach drastically improves clock-cycle efficiency and overcomes the memory wall bottleneck inherent in massive Transformer workloads.

---

## Documentation
For a comprehensive overview of the system's architecture, the cNoC execution mode, routing protocols (Serpentine Sort), and a detailed technical reference of the underlying C++ classes, please refer to the official documentation:

**[NoCDAS Documentation: A Transition to cNoC for LLM Inference](./doc/Documentation.pdf)**

For a more practical, hands-on guide covering simulation setup, workflows, and expected results when using LLMs and the cNoC paradigm, please check out our Toy Model guide:

**[Practical Guide to LLM & cNoC Simulation](./doc/toy_model.pdf)**

---

## Key Features
*   Dual Execution Paradigms:
    *   *Baseline Mode:* Traditional Data-to-Processor execution (standard approach for CNNs and baseline execution).
    *   *cNoC Mode:* Highly optimized In-Transit Computation paradigm. Turns NoC routers into active processing units via Multi-way Function Units (MFUs) to maximize clock-cycle efficiency during LLM inference.
*   Hardware-Aware Memory Modeling: Simulates local SRAM constraints, dedicated PE-level KV-Caches, and Router-level weights.
*   Dynamic Quantization & Scaling: Features an `INT8_QUANTIZATION` toggle that employs a "Fake-Quantization" approach. It realistically scales effective SRAM capacity and reduces NoC payload traffic by 4x to simulate the physical footprint of 8-bit hardware accelerators, without compromising the underlying FP32 mathematical simulation.
*   Advanced LLM Support: Implements Rolling KV-Cache logic for infinite-length sequence generation and in-network online softmax reduction.
*   Cycle-Accurate Fidelity: Models physical hardware constraints including buffer depths, link bandwidths, and specific Special Function Unit (SFU) latencies.

## Environment
*   C++ Version: C++14 or above.
*   Build System: Support for CMake (>= 3.10) on Ubuntu 20.04.
*   Tested Environments: Eclipse CDT project on Windows 10 / Ubuntu 20.04.

## Quick Start & Examples

The simulator defaults to running a baseline CNN (LeNet) on an 8x8 NoC with random mapping. 

The `input/` folder contains various workloads:
*   LeNet: Includes model, input, and weight files for both FE and RE modes.
*   AlexNet & DarkNet: Model files available for RE mode. *(Tip: For DarkNet, run `darknet1.txt` with a 1/16 sampling of the original IFMap size to shorten simulation time).*
*   Transformer/LLM Workloads: Load LLM specific text files. For optimal clock-cycle efficiency, enable `cNoC_MODE` in the configuration.

## Configuration
The simulator's behavior and hardware specifications are centrally managed in `src/parameters.hpp`.

*   Topology & NoC Nodes: Change NoC dimensions and memory node placement (e.g., `MemNode8`, `X_NUM`, `Y_NUM`).
*   Workload Selection: Modify model file names directly in the header or pass them via command-line arguments.
*   Hardware Optimizations: Toggle macros like `cNoC_MODE`, `ENABLE_KV_CACHE`, and `ROUTER_SRAM_LIMIT` to enable active NoC processing for LLMs.
*   Quantization Scaling: Use `INT8_QUANTIZATION` to dynamically scale physical memory constraints and simulate the bandwidth savings of INT8 hardware.
*   Logging: Enable the `Countlatency` macro for detailed output logs and performance metrics.

## Input Model Format
The simulator parses model architectures from text files. Supported layers include both standard CNN operations and advanced LLM/Transformer opcodes:

**CNN / Baseline Layers:**
*   `Input` *size_x size_y size_channel*
*   `Conv2D` *in_channel kernel_x kernel_y out_channel activation padding stride*
*   `Pool` *in_channel kernel_x kernel_y out_channel padding stride* (Max Pooling)
*   `AvgPool` *in_channel kernel_x kernel_y out_channel padding stride*
*   `Dense` *in_size out_size activation*

*(Activation options: "relu", "tanh", "sigmoid", "linear", "swiglu", "geglu")*

**Transformer / LLM Layers:**
*   `MATMUL` - Standard matrix multiplication (Query/Key/Value/FFN projections).
*   `RMSNORM` / `LAYERNORM` - Normalization layers.
*   `SOFTMAX_TR` - Causal masked softmax for self-attention.
*   `ATTENTION` - Fused hardware attention logic.
*   `ROPE` - Rotary Positional Embeddings.
*   `EMBEDDING` - Vocabulary lookup operations.
*   `ADD` - Residual skip connections.

## Memory Controller Placement Configuration
Guidance on customizing Memory Controller (MC) placements:

1.  Select Topology: In `parameters.hpp`, activate or create a `MemNode` Macro. Assign the number of cores (`PE_X_NUM`, `PE_Y_NUM`) and routers in the NoC (`X_NUM`, `Y_NUM`, `TOT_NUM`).
2.  Define Router IDs: In `MAC.hpp`, define the position of MC cores by their connected router ID. 
    *   *Example:* `const int dest_list[] = {17, 18}` means MC cores are connected to routers 17 and 18.
3.  Assign PEs to MCs: In `MAC.cpp` (within the `MAC::MAC` initialization function), define the destination MCs (`dest_mem_id`) for each PE. This can be done via direct or rule-based assignment. 
    *   *Example:* `if (xid <= 3) { dest_mem_id = dest_list[0]; } else { dest_mem_id = dest_list[1]; }`
4.  Build & Run: Recompile the simulator to apply the new MC placement configuration.

## Build & Run
You can compile and run the simulator using either the provided shell script or manually via CMake.

**Option 1: Quick Start**

The easiest way to compile and execute the simulator is by running the bash script from the root directory:
```bash
./run.sh
```

**Option 2: Manual CMake Build**

If you need to recompile from scratch or prefer a manual build process, execute the following commands:

```bash
rm -rf build
mkdir build && cd build
cmake ..
make
cd ..
./build/NoCDASim
```

## Reference
If you use NoCDAS in your research, please cite the following paper:

> Wenyao Zhu, Yizhi Chen, and Zhonghai Lu, “NoCDAS: A Cycle-Accurate NoC-Based Deep Neural Network Accelerator Simulator”, *ACM Transactions on Modeling and Computer Simulation*, April 2025. [https://doi.org/10.1145/3729169](https://doi.org/10.1145/3729169) (Open access)