/*
 * flit_buffer.cpp
 *
 */

#include <cassert>
#include "FlitBuffer.hpp"
#include "../parameters.hpp"
#include <iostream>

extern unsigned int cycles;

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
  
  if (credit_delay > 0) {
    credit_return_queue.push_back(cycles + credit_delay);
  } else {
    used_credit--;
  }
  
  return t_flit;
}

void FlitBuffer::update_credits() {
  while(!credit_return_queue.empty() && credit_return_queue.front() <= cycles) {
    credit_return_queue.pop_front();
    used_credit--;
  }
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
  for(auto flit : flit_queue) {
    Flit::release(flit);
  }
  flit_queue.clear();
  cur_flit_num = 0;
  used_credit = 0;
  credit_return_queue.clear();
}

bool FlitBuffer::isFull(){
  update_credits();
  return ( (depth<=used_credit) ? 1:0);
}

FlitBuffer::~FlitBuffer(){
  Flit* flit;
  while(flit_queue.size()!=0){
    flit = flit_queue.front();
    flit_queue.pop_front();
    Flit::release(flit);
  }
}



