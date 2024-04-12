/*
 * MAC.hpp
 * Describe PE functions.
 * Message insert.
 */

#ifndef MAC_HPP_
#define MAC_HPP_

#include <vector>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <deque>
#include <cmath>
#include <cassert>
#include "parameters.hpp"
#include "NoC/Packet.hpp"
#include "NoC/NI.hpp"
#include "MACnet.hpp"

//
//#define MEM_NODES 4
//const int dest_list[] = {18, 21, 42, 45};


// memory node (NI)

#ifdef MemNode4
	#define MEM_NODES 4
	const int dest_list[] = {18, 21, 42, 45}; // 8*8
#elif defined MemNode8
	#define MEM_NODES 8
	const int dest_list[] = {17, 18, 21, 22, 41, 42, 45, 46};  // 8*8
#elif defined MemNode2
	#define MEM_NODES 2
	const int dest_list[] = {9, 10}; // 4*4
#elif defined MemNode18
	#define MEM_NODES 18
	const int dest_list[] = {25,26,29,30,33,34,73,74,77,78,81,82,121,122,125,126,129,130};  // 12*12
#elif defined MemNode32
	#define MEM_NODES 32
	const int dest_list[] = {33,34,37,38,41,42,45,46,97,98,101,102,105,106,109,110,161,162,165,166,169,170,173,174,225,226,229,230,233,234,237,238};  // 12*12
#elif defined MemNode5
	#define MEM_NODES 5
	const int dest_list[] = {13, 14, 17, 25, 28};  // 6*6
#elif defined MemNode13
	#define MEM_NODES 13
	const int dest_list[] = {21,22,25,26,29,61,62,65,66,69,81,85,88};  // 10*10
#endif
//

using namespace std;

extern int packet_id;

extern unsigned int cycles;

extern vector<vector<int>> DNN_latency;

class MACnet;
class Packet;

class MAC
{
	public:
  /** @brief MAC
   *
   */
	MAC (int t_id, MACnet* t_net, int t_NI_id);
	MACnet* net;
	int id;
	int fn;
	int pecycle;
	int selfstatus;
	int request;
	int tmp_request;

	int send;
	int NI_id;
	deque<float> weight;
	deque<float> infeature;
	deque<float> inbuffer;
	int ch_size;
	int m_size;
	int dest_mem_id;
	int tmpch;
	int tmpm;
	int m_count;
	float outfeature{}; //from MRL
	deque <int> routing_table;

	// for new pooling
	int npoolflag;
	int n_tmpch;
	deque<int> n_tmpm;


	MAC* nextMAC;

	bool inject (int type, int d_id, int data_length, float t_output, NI* t_NI, int p_id, int mac_src);
	void receive (Message* re_msg);
	void runOneStep();
	void sigmoid(float& x);
	void tanh(float& x);
	void relu(float& x);


	~MAC ();
};




#endif /* MAC_HPP_ */
