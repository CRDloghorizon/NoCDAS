/*
 * PacketBuffer.cpp
 *
 */

#include "PacketBuffer.hpp"
#include <cassert>

PacketBuffer::PacketBuffer (NI* t_NI, int t_vn_num)
{
  packet_num = 0;
  vn_num = t_vn_num;
  ni_owner = t_NI;
}

void PacketBuffer::enqueue(Packet* t_packet){
  packet_queue.push_back(t_packet);
  packet_num++;
}

Packet* PacketBuffer::dequeue(){
  assert(packet_num!=0);
  Packet* packet = packet_queue.front();
  packet_queue.pop_front();
  packet_num--;
  return packet;
}

Packet * PacketBuffer::read(){
  assert(packet_num!=0);
  return packet_queue.front();
}

PacketBuffer::~PacketBuffer(){
  Packet* packet;
  while(packet_queue.size()!=0){
      packet = packet_queue.front();
      packet_queue.pop_front();
      delete packet;
  }
}

