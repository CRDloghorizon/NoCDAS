# NoCDAS
A Cycle-Accurate NoC-based Deep Neural Network Accelerator Simulator

## Environment
C++14 or above.
Tested as Eclipse CDT project on Windows 10 / Ubuntu 20.04

## Example
The default setup runs a LeNet on an 8x8 NoC with random mapping.

## Simulation
The input folder gives the LeNet model file, input file, and weight file for both FE and RE modes.
It also gives the AlexNet and DarkNet model files for RE mode. 
For DarkNet, run "darknet1.txt" with a 1/16 sampling of the original IFMap size to shorten the running time.
