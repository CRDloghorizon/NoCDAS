/*
 * NI.hpp
 *
 */

#ifndef NI_HPP_
#define NI_HPP_

#include "FlitBuffer.hpp"
#include "Link.hpp"
#include "PacketBuffer.hpp"
#include "ROutPort.hpp"
#include "VCNetwork.hpp"
#include "NRBase.hpp"
#include "../parameters.hpp"

#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>

extern unsigned int cycles;
extern vector<vector<int>> DNN_latency;

class FlitBuffer;
class VCNetwork;
class RInPort;
class PacketBuffer;
class ROutPort;
class Flit;

//added packet id
extern int packet_id;
extern std::vector<std::vector<int>> LCS_packet_delay;
extern std::vector<std::vector<int>> URS_packet_delay;
extern std::ofstream flit_trace;

class NI: public NRBase
{
public:
  NI (int t_id, VCNetwork* t_vcNetwork, int t_vn_num, int t_vc_per_vn, int t_vc_priority_per_vn, int t_in_depth, int* t_NI_num);

  int* NI_num;
  int id;
  VCNetwork* vcNetwork;
  int vn_num;
  int vc_per_vn;
  int vc_priority_per_vn;

  // added
  unsigned int NI_cycle_npooling;


  std::vector<PacketBuffer*> packetBuffer_list;
  std::vector<std::deque<Packet*> > packet_buffer_out;  // 0 request; 1 response
  std::vector<FlitBuffer*> buffer_list;
  std::vector<int> out_vc;
  std::vector<int> state; // 0 I; 1 V; 2 A;

  std::vector<int> priority_vc;
  int count_vc; // starvation forbidden
  std::vector<int> priority_switch;
  int count_switch; // starvation forbidden

  int rr_buffer;
  int in_depth;

  // for individual priority
  int starvation;
  int rr_priority_record;

  ROutPort* out_port;

  RInPort* in_port;

  static int count_s, count_r;
  static int count_s_resp, count_r_resp;

  static int count_input;

  //added
  int num_flit;

  int total_delay;
  int total_num;
  int total_delay_URS;
  int total_num_URS;

  static int worst_LCS;
  static int worst_URS;

  static int URS_delay_distribution[DISTRIBUTION_NUM];
  static int LCS_delay_distribution[DISTRIBUTION_NUM];

  static int URS_latency_single_dis[100];
  static int URS_latency_single_count;
  static int URS_latency_single_total;
  static int URS_latency_single_worst;

  static int LCS_latency_single_dis[100];
  static int LCS_latency_single_count;
  static int LCS_latency_single_total;
  static int LCS_latency_single_worst;

  static int converse_latency_single_dis[100];
  static int converse_latency_single_count;
  static int converse_latency_single_total;
  static int converse_latency_single_worst;



  // send
  bool flitize(Packet*, int);

  void packetDequeue();
  void vcAllocation();
  void switchArbitration();
  void dequeue();

  // receive

  void inputCheck();

  // main
  void runOneStep();

  ~NI();

};

#endif /* NI_HPP_ */
