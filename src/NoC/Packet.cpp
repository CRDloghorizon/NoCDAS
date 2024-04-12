/*
 * Packet.cpp
 *
 */

#include "Packet.hpp"
#include <iostream>
//type convert: add change type and vnet

Packet::Packet(Message t_message, int router_num_x, int* NI_num){
  this->message = t_message;
  int t_type = message.type; //0-> read request (93 bits); 1-> read response (6+y bits); 2-> write request (93+y bits); 3-> write response (6 bits);
  int data_length = message.data_length; // in bytes now (float *4 byte) 	//in bits <y bits>
  dest_convert(message.destination, router_num_x, NI_num);
  switch (t_type){
    case 0: length = data_length*2 + 2; //length = 1 int
            type = 0;
            vnet = 0;
            break;
    case 1: length = data_length*2 + 2; //length = n float (fp16)
            type = 0;	// changed case 1 and 2
            vnet = 0;
            break;
    case 2: length = data_length*2 + 2; //length = 1 float;  overhead may increase here later
            type = 1;
            vnet = 0;
            break;
    case 3: length = 1 + 2; // length = 3byte (1byte confirm + 2byte overhead)
            type = 1;
            vnet = 0;
            break;
  }
  send_out_time = 0;
  // added
  in_net_time = 0;
}

/*
  * @brief convert destination to be x/y/output port of the router
   */
void Packet::dest_convert(int dest, int router_num_x, int* NI_num){
  int hist_num = 0, cur_num = 0, router = 0;
  while (cur_num <= dest){
      hist_num = cur_num;
      cur_num += NI_num[router];
      router++;
  }
  router--;
  destination[0] = router/router_num_x;
  destination[1] = router%router_num_x;
  destination[2] = dest-hist_num;
  //std::cout << destination[0] <<destination[1] << destination[2]<<std::endl;
}


