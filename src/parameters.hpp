/*
 * parameters.hpp
 * Configuration file for NoCDAS
 */

#ifndef PARAMETERS_HPP_
#define PARAMETERS_HPP_

#define DEFAULT_NNMODEL_FILENAME	"./src/input/lenet.txt"   //lenet alexnet newnet0 newnet vgg: p, p4, p5
#define DEFAULT_NNWEIGHT_FILENAME	"./src/input/weight.txt"
#define DEFAULT_NNINPUT_FILENAME	"./src/input/input2.txt"


/******************************/
// define evaluation modes
#define fulleval
//#define randomeval


#define only3type	//only 3 packets per neuron task
//#define newpooling	//new pooling in router near memory
#define outPortNoInfinite // back pressure from VC Router Out Port
//#define Countlatency		// open recording of packet level delay

/*******************************/
#define MemNode2  // 2 MC cores (for 4*4)
//#define MemNode8  // 8 MC cores (for 8*8)
//#define MemNode18  // 18 MC cores (for 12*12)
//#define MemNode32  // 32 MC cores (for 16*16)

//#define MemNode5  // 5 MC cores (for 6*6)
//#define MemNode13  // 13 MC cores (for 10*10)

//#define MemNode4  // 4 MC cores 8by8

/******************************/
//#define rowmapping
//#define colmapping
#define randmapping

/******************************/
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
#define FREQUENCY 2 // GHz (NoC)
#define PE_NUM_OP 25   //25 OP per PE cycle
#define PE_FREQ_RATIO 10 // NoC freq / PE freq, PE 200MHz -> 10
#define MEM_read_delay 0.3125 // delay for 2byte / 1 data
#define POOLING_DELAY 10 // delay for new pooling on the go 4 + 2 + 2 + 2
//////////////////////////////////
#define VN_NUM 1   //2
#define VC_PER_VN  4  ///<1 A: control URS (control all in other 3 modes)
#define VC_PRIORITY_PER_VN 0 ///< B: only control LCS
#define STARVATION_LIMIT 20 // forbid starvation (no priority packet must go after 20)
#define LCS_URS_TRAFFIC
#define INPORT_FLIT_BUFFER_SIZE 4; // number 4
#define FLIT_LENGTH 32 // byte 32*8=256bit  //32
#define INFINITE 10000    // added for in port and out port buffer
#define INFINITE1 10000  // added for flit buffer (10000)
#define CACHE_DELAY 1  // simulate cache memory access delay (5ns = 1 clock cycle under 200 MHz frequency)
#define LINK_TIME 2
#define DISTRIBUTION_NUM 20

#define PRINT 1000000 // define the prining steps in cycle

struct GlobalParams {
	static char NNmodel_filename[128];
	static char NNweight_filename[128];
	static char NNinput_filename[128];
};


#endif /* PARAMETERS_HPP_ */
