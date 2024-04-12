/*
 * flit.cpp
 *
 */

#include "Flit.hpp"
#include "../parameters.hpp"

const int Flit::length = FLIT_LENGTH;

Flit::Flit(int t_id, int t_type, int t_vnet, int t_vc, Packet* t_packet, float t_cycles, int t_pid){
  id = t_id;
  type = t_type;
  vnet = t_vnet;
  vc = t_vc;
  out_port = -1;
  packet = t_packet;
  sched_time = t_cycles;
  packetid=t_pid;
}




