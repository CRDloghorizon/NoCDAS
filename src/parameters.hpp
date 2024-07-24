/*
 * parameters.hpp
 * Configuration file for NoCDAS
 */

#ifndef PARAMETERS_HPP_
#define PARAMETERS_HPP_

#define DEFAULT_NNMODEL_FILENAME	"./src/input/lenet.txt"   // the path to the DNN model file, can be changed in line argument
#define DEFAULT_NNWEIGHT_FILENAME	"./src/input/weight.txt"  // the path to the DNN weight file, can be changed in line argument
#define DEFAULT_NNINPUT_FILENAME	"./src/input/input2.txt"  // the path to the DNN input file, can be changed in line argument


/******************************/
// Here we define evaluation modes. DNN model file is always required. In FE mode, weight and input files are required.
#define fulleval	// FE mode
//#define randomeval	//RE mode


#define only3type	// 3 packets per neuron task, otherwwise 4 packets.
//#define newpooling	// Pooling in the network optimization switch
#define outPortNoInfinite // Simulate back pressure from VC Router Out Port

/******************************/
// Trace recording. The detailed explaination is in /output/recorded.txt
//#define Countlatency		// Open recording of packet level trace and latency
//#define CountNum 30000    // Set the maximum number of output packet traces

/******************************/
// NoC Node configuration macros used in the manuscript
#define MemNode2  // 2 MC cores (for 4*4 NoC)
//#define MemNode8  // 8 MC cores (for 8*8 NoC)
//#define MemNode18  // 18 MC cores (for 12*12 NoC)
//#define MemNode32  // 32 MC cores (for 16*16 NoC)

//#define MemNode5  // 5 MC cores (for 6*6 NoC)
//#define MemNode13  // 13 MC cores (for 10*10 NoC)

//#define MemNode4  // 4 MC cores (for 8*8 NoC)

/******************************/
// Task mapping macros
//#define rowmapping	//row-major mapping
//#define colmapping	//column-major mapping
#define randmapping		// random mapping

/******************************/
// Detailed NoC size definitions for Node configuration macros 
#ifdef MemNode2
	#define PE_X_NUM 4
	#define PE_Y_NUM 4
	//NI size
	#define X_NUM 4
	#define Y_NUM 4
	#define TOT_NUM 16
#elif defined MemNode4
// PE_Y_NUM (mac size) should be several times of NI TOT_NUM
	#define PE_X_NUM 8
	#define PE_Y_NUM 8
	//NI size
	#define X_NUM 8
	#define Y_NUM 8
	#define TOT_NUM 64
#elif defined MemNode8
	#define PE_X_NUM 8
	#define PE_Y_NUM 8
	//NI size
	#define X_NUM 8
	#define Y_NUM 8
	#define TOT_NUM 64
#elif defined MemNode18
	#define PE_X_NUM 12
	#define PE_Y_NUM 12
	//NI size
	#define X_NUM 12
	#define Y_NUM 12
	#define TOT_NUM 144
#elif defined MemNode32
	#define PE_X_NUM 16
	#define PE_Y_NUM 16
	//NI size
	#define X_NUM 16
	#define Y_NUM 16
	#define TOT_NUM 256
#elif defined MemNode5
	#define PE_X_NUM 6
	#define PE_Y_NUM 6
	//NI size
	#define X_NUM 6
	#define Y_NUM 6
	#define TOT_NUM 36
#elif defined MemNode13
	#define PE_X_NUM 10
	#define PE_Y_NUM 10
	//NI size
	#define X_NUM 10
	#define Y_NUM 10
	#define TOT_NUM 100
#endif


//////////////////////////////////
#define FREQUENCY 2 // GHz (NoC clock frequency)
#define PE_NUM_OP 25   // 25 OP per PE cycle (define the MAC array size in PE)
#define PE_FREQ_RATIO 10 // NoC freq / PE freq, PE 200MHz -> 10 (define the nodes (PE/MC) clock frequency)
#define MEM_read_delay 0.3125 // delay for 2byte / 1 data (define the cache data transfer speed)
#define POOLING_DELAY 10 // define extra delay for pooling in the network 4 + 2 + 2 + 2 clock cycles
//////////////////////////////////
#define VN_NUM 1   // number of virtual networks, here we use 1 means only one virtual network on the real NoC hardware
#define VC_PER_VN  4  // number of VC channels per port in one nerwork
#define VC_PRIORITY_PER_VN 0 // define priority VC channels in network
#define STARVATION_LIMIT 20 // forbid starvation (no priority packet must go after 20)
#define LCS_URS_TRAFFIC		// standard traffic mode, LCS: Latency Critical Service, URS: Unspecified Rate Service
#define INPORT_FLIT_BUFFER_SIZE 4; // flit buffer size per VC channel
#define FLIT_LENGTH 32 // byte 32*8=256bit  // lenghth of flit in byte
#define INFINITE 10000    // added for in port and out port buffer
#define INFINITE1 10000  // added for flit buffer
#define CACHE_DELAY 1  // simulate cache memory access delay (5ns = 1 clock cycle under 200 MHz frequency)
#define LINK_TIME 2		// latency for link transfer
#define DISTRIBUTION_NUM 20	// threshold for counting end-to end packet delay is good (<20 is good)

#define PRINT 1000000 // define the prining steps in clock cycle

// reserve string for input file paths
struct GlobalParams {
	static char NNmodel_filename[128];
	static char NNweight_filename[128];
	static char NNinput_filename[128];
};


#endif /* PARAMETERS_HPP_ */
