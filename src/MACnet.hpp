/*
 * MACnet.hpp
 * Global control and MC functions
 */

#ifndef MACNET_HPP_
#define MACNET_HPP_

#include <cmath>
#include <vector>
#include <deque>
#include <iostream>
#include <fstream>
#include <algorithm>    //std::shuffle
#include <random>       // std::default_random_engine
#include <chrono>       // std::chrono::system_clock
#include "MAC.hpp"
#include "Model.hpp"
#include "NoC/VCNetwork.hpp"

using namespace std;

extern int packet_id;

extern unsigned int cycles;

extern vector<vector<int>> DNN_latency;

// NoC
class VCNetwork;

class MAC;

class MACnet
{
public:

  /** @brief MAC Network
   *
   */
	MACnet (int mac_num, int t_pe_x, int t_pe_y, Model *m, VCNetwork* t_Network);
	std::vector<MAC*> MAC_list;
	VCNetwork* vcNetwork;

	void create_input();
	vector<vector<float>> weight_table;
	vector<vector<float>> input_table;
	vector<vector<float>> output_table;

	deque< deque< int > > mapping_table;
	void mapping(int neuronnum);
	void ymapping(int neuronnum);
	void rmapping(int neuronnum);

	void runOneStep();
	void checkStatus();

	Model* cnnmodel;
	int macNum;
	int pe_x;
	int pe_y;
	int used_pe;

	int c_layer; // current layer
	int n_layer; // total layer

	int in_ch; // for input channel
	int in_x;
	int in_y;

	// for new pooling
	int no_x;
	int no_y;
	int nw_x;
	int nw_y;
	int no_ch; // next o_ch in pooling layer
	int npad; // padding
	int nstride; // stride
	//vector<vector<float>> pooling_table;


	int w_ch; // for filter
	int w_x; 
	int w_y;
	int st_w;
	int pad;
	int stride;

	int o_ch; // for output
	int o_x; 
	int o_y;

	int o_fn; // for function

	int readyflag;

	// for print
	vector<int> Layer_latency;

	~MACnet ();
};



#endif /* MACNET_HPP_ */
