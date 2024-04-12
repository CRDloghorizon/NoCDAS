/*
 * flit_buffer.cpp
 *
 */

#include <cassert>
#include "FlitBuffer.hpp"
#include "../parameters.hpp"
#include <iostream>


FlitBuffer::FlitBuffer(int t_vc, int t_vnet, int t_id, int t_depth){
  id = t_id;
  vnet = t_vnet;
  vc = t_vc;
  cur_flit_num = 0;
  used_credit = 0;
  depth = t_depth;
}

Flit* FlitBuffer::read(){
  assert(cur_flit_num != 0);
  return flit_queue.front();
}


Flit* FlitBuffer::dequeue(){
  assert (cur_flit_num != 0);
  Flit* t_flit = flit_queue.front();
  flit_queue.pop_front();
  cur_flit_num--;
  used_credit--;
  return t_flit;
}

void FlitBuffer::get_credit(){
  used_credit++;
}

void FlitBuffer::enqueue(Flit* t_flit){
	if(cur_flit_num == depth){
		cout<<" depth is "<<depth<<endl;
	}
	assert (cur_flit_num != depth);
	flit_queue.push_back(t_flit);
	cur_flit_num++;
}

Flit* FlitBuffer::readLast(){
  assert(cur_flit_num != 0);
  return flit_queue.back();
}

void FlitBuffer::empty(){
  flit_queue.empty();
  cur_flit_num = 0;
  used_credit = 0;
}

bool FlitBuffer::isFull(){
  return ( (depth<=used_credit) ? 1:0);
}

FlitBuffer::~FlitBuffer(){
  Flit* flit;
  while(flit_queue.size()!=0){
      flit = flit_queue.front();
      flit_queue.pop_front();
      delete flit;
  }
}



