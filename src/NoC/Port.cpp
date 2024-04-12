/*
 * Port.cpp
 *
 */

#include "Port.hpp"
#include <iostream>
#include "../parameters.hpp"


Port::Port (int t_id, int t_vn_num, int t_vc_per_vn, int t_vc_priority_per_vn, int t_depth)
{
   id = t_id;
   vn_num = t_vn_num;
   vc_per_vn = t_vc_per_vn;
   vc_priority_per_vn = t_vc_priority_per_vn;

   depth = t_depth;
   buffer_list.reserve(vn_num*(vc_per_vn + vc_priority_per_vn));

   //added
   count_flit = 0;

   // URS 0~vn_num*vc_per_vn-1;
   for(int i=0; i<vn_num*vc_per_vn; i++){
       FlitBuffer * t_flitBuffer = new FlitBuffer(i%vc_per_vn, i/vn_num, i, depth);
       buffer_list.push_back(t_flitBuffer);
   }

   // LCS vn_num*vc_per_vn~vn_num*(vc_per_vn + vc_priority_per_vn)-1
   for(int i=0; i<vn_num*vc_priority_per_vn; i++){
       FlitBuffer * t_flitBuffer = new FlitBuffer(i%vc_priority_per_vn+vc_per_vn, i/vn_num, i, depth);
       buffer_list.push_back(t_flitBuffer);
   }
}


Port::~Port(){
  FlitBuffer* flitBuffer;
  while(buffer_list.size()!=0){
      flitBuffer = buffer_list.back();
      buffer_list.pop_back();
      delete flitBuffer;
  }
}



