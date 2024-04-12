/*
 * PacketBuffer.hpp
 *
 */

#ifndef VC_PACKETBUFFER_HPP_
#define VC_PACKETBUFFER_HPP_

#include "NI.hpp"
#include "Packet.hpp"
#include <deque>


class NI;


class PacketBuffer
{
public:
  PacketBuffer (NI* ni_owner, int id);

  std::deque<Packet*> packet_queue;

  void enqueue(Packet*);
  Packet* dequeue();
  Packet * read();
  ~PacketBuffer();

  int packet_num;
  int vn_num;

  NI* ni_owner;

};

#endif /* VC_PACKETBUFFER_HPP_ */
