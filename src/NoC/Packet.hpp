/*
 * Packet.hpp
 *
 */

#ifndef PACKET_HPP_
#define PACKET_HPP_

#include <vector>
#include <stdio.h>
#include <deque>
using namespace std;


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
  deque<float> data;
  //for pooling
  // int poutid;	// for pooling table check
  int penable;  // 0 no, 1 max, 2 avg

};


class Packet
{
public:
  Packet(Message t_message, int router_num_x, int* NI_num);

  Message message;
  int length;  // length in byte
  int type; // 0 -> request; 1 -> response;
  int vnet;
  int destination[3];  // x, y, output port of the router; from 0 ..

  void dest_convert(int dest, int router_num_x, int* NI_num);

  int send_out_time;

  //added
  int in_net_time;



};


#endif /* PACKET_HPP_ */
