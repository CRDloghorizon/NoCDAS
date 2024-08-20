# NoCDAS
A Cycle-Accurate NoC-based Deep Neural Network Accelerator Simulator

## Environment
C++14 or above.
Tested as Eclipse CDT project on Windows 10 / Ubuntu 20.04.

Also support CMake (>= 3.10) on Ubuntu 20.04.

## Example
The default setup runs a LeNet on an 8x8 NoC with random mapping.

## Simulation

The input folder gives the LeNet model file, input file, and weight file for both FE and RE modes.
It also gives the AlexNet and DarkNet model files for RE mode. 
For DarkNet, run "darknet1.txt" with a 1/16 sampling of the original IFMap size to shorten the running time.

## Configuration

By changing the NoC node configuration macros in src/parameters.hpp, user can test on different NoCs.

By changing the model file names in src/parameters.hpp or through line arguments, user can test on different DNN workloads.

Detailed output logs can be enabled by the 'Countlatency' macro in src/parameters.hpp.

## Input Model Format

Input layer: Input size_x size_y size_channel 

Convolution layer: Conv2D in_channel kernel_x kernel_y out_channel activation padding stride 

Max pooling layer: Pool in_channel kernel_x kernel_y out_channel padding stride 

Average pooling layer: AvgPool in_channel kernel_x kernel_y out_channel padding stride

Fully connected layer: Dense in_size out_size activation

//"activation" is the non-linear activation function: "relu"/"tanh"/"sigmoid"/"linear"

## MC Placement Configuration

Guidance on changing MC placements:

1. In parameters.hpp, activate or create a MemNode Macro. Assign the number of cores (PE_X_NUM, PE_Y_NUM) and routers in NoC (X_NUM, Y_NUM, TOT_NUM).

2. In MAC.hpp, at the beginning of this header file, define the position of MC cores by router ID. For example: const int dest_list[] = {17, 18} means that MC cores are connected to router ID 17, 18.

3. In MAC.cpp, in the initialization function MAC::MAC, define the destination MCs (dest_mem_id) for each PE. This can be done by direct assignment or rule-based assignment. For example:
  if (xid <= 3) {dest_mem_id = dest_list[0];} else {dest_mem_id = dest_list[1];}

4. Recompile the simulator and run with the new MC placement configuration.
