/*
 * Packet.hpp
 *
 */

#ifndef PACKET_HPP_
#define PACKET_HPP_

#include <vector>
#include <stdio.h>

struct Message{
  int NI_id;
  int mac_id;
  int out_cycle;
  int slave_id;
  int sequence_id;
  int type;
  int data_length;
  int destination;
  int QoS = 0;
  int source_id;
  int signal_id;
  
  std::vector<float> data;          // Contains inputs and partial sums
  int psum_offset = 0;              // Contains the start index of psum in 'data'.
  int k_dim = 0;

  double running_max = -1e9;        // Tracks the maximum score (m)
  double running_sum = 0.0;         // Tracks the sum of exponentials (l)

  int compute_op;                   // type of operation (es. MATMUL, ADD)
  
  std::vector<int> routing_path;    // Source routing: sorted list of routers ID to traverse
  
  //for pooling
  int penable;                       // 0 no, 1 max, 2 avg
};

class Packet
{
public:
  Packet(Message t_message, int router_num_x, int* NI_num);

  Message message;
  int length;                       // byte length
  int type;                         // 0 -> request; 1 -> response; 4 -> distribution; 5 -> computation;
  int vnet;
  int destination[3];               // x, y, output port of the router

  float send_out_time;              // time of packet sent from PE
  float in_net_time;                // time of packet insert in to the NoC

  void dest_convert(int dest, int router_num_x, int* NI_num);
  int get_next_router_dest();       // source routing helper

  int current_path_index;           // Tracks progress in Source Routing

  // --- Object Pool ---
  static std::vector<Packet*> free_pool;
  static Packet* allocate(Message t_message, int router_num_x, int* NI_num);
  static void release(Packet* packet);
  void reset(Message t_message, int router_num_x, int* NI_num);
};

#endif /* PACKET_HPP_ */