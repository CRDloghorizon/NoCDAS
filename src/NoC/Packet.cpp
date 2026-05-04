/*
 * Packet.cpp
 *
 */

#include "Packet.hpp"
#include "../parameters.hpp"
#include <iostream>
//type convert: add change type and vnet

Packet::Packet(Message t_message, int router_num_x, int* NI_num) 
    : message(std::move(t_message))
{
  int t_type = message.type; 
  int data_length = message.data_length;
  
  current_path_index = 0;

  dest_convert(message.destination, router_num_x, NI_num);
  
  switch (t_type){
    case 0: length = data_length * DATA_BYTES + 2;
            type = 0;
            vnet = 0;
            break;
    case 1: length = data_length * DATA_BYTES + 2;
            type = 0;	
            vnet = 0;
            break;
    case 2: length = data_length * DATA_BYTES + 2;
            type = 1;
            vnet = 0;
            break;
    case 3: length = 1 + 2; 
            type = 1;
            vnet = 0;
            break;
    case 4: // PACKET_DISTRIBUTION
            length = data_length * DATA_BYTES + 2; // weights are float (4 byte)
            type = 0;
            vnet = 1;
            break;
    case 5: // PACKET_COMPUTATION
            // The length includes both the input data (es. input_vector) and the space for the psum
            // Length depends on the unified buffer 'data' only.
            length = message.data.size() * DATA_BYTES + 4; // +4 bytes overhead per opcode
            type = 0; // Like request but with higher priority
            vnet = 1;
            break;
  }

  // Flit Alignment and Padding
  // Hardware transfers data in fixed-size flits. We must round up the raw length 
  // to the nearest multiple of FLIT_LENGTH to ensure cycle-accurate simulation.
  // We use integer math: ((length + FLIT_LENGTH - 1) / FLIT_LENGTH) * FLIT_LENGTH
  
  if (length % FLIT_LENGTH != 0) {
      length = ((length + FLIT_LENGTH - 1) / FLIT_LENGTH) * FLIT_LENGTH;
  }

  send_out_time = 0;
  in_net_time = 0;
}

// Source Routing helper
int Packet::get_next_router_dest() {
    if (current_path_index < message.routing_path.size()) {
        return message.routing_path[current_path_index];
    }
    return -1;
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


