/*
 * flit_buffer.hpp
 *
 */

#ifndef VC_FLITBUFFER_HPP_
#define VC_FLITBUFFER_HPP_

#include "Flit.hpp"
#include <deque>

using namespace std;

class FlitBuffer{
public:
  FlitBuffer(int t_vc, int t_vnet, int t_id, int t_depth);

  Flit* read();
  Flit* readLast();
  void enqueue(Flit* flit);
  Flit* dequeue();
  void empty();
  bool isFull();
  void get_credit();

  ~FlitBuffer();

  int id;
  int vc;
  int vnet;
  int depth;
  int cur_flit_num;
  int used_credit;
  deque <Flit*> flit_queue;
};




#endif /* VC_FLITBUFFER_HPP_ */
