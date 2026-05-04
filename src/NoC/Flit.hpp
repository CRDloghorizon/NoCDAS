/*
 * flit.hpp
 *
 */

#ifndef VC_FLIT_HPP_
#define VC_FLIT_HPP_

#include "Packet.hpp"
#include <vector>
#include <bitset>

class Packet;

class Flit{
public:


  /*
   * @brief set values to be the same as input
   *  id = t_id;
      type = t_type;
      vnet = t_vnet;
      vc = t_vc;
      out_port = -1;
      packet = t_packet;
      sched_time = t_cycles;
   */
  Flit(int t_id, int t_type, int t_vnet, int t_vc, Packet* t_packet, float t_cycles, int t_pid);

  const static int length;  // length in byte
  int id;   // The sequence id in a packet
  int type; // 0 -> head; 1 -> tail; 2 -> body; 10 -> head_tail;
  int vnet;
  int vc;
  int out_port;
  int packetid;
  
  float sched_time; // if sched_time < cur_time, then the flit can be transferred.
  // added To trace the passing node id and cycles
  std::vector<int> trace_node;
  std::vector<int> trace_time;

  std::vector<bool>computed_routers;

  int global_data_offset;

  int current_payload_size;
  
  float get_data(int local_index) const;
  int get_payload_size() const;

  void update_data(int local_index, float new_val);

  Packet * packet;
};



#endif /* VC_FLIT_HPP_ */
