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
    
    computed_routers.resize(TOT_NUM, false);

    current_payload_size = 0;

    if (packet != nullptr && (packet->message.type == 4 || packet->message.type == 5)) {
        // int floats_per_flit = FLIT_LENGTH / 4; 
        int elements_per_flit = FLIT_LENGTH / DATA_BYTES;
        int op = packet->message.compute_op;
        
        global_data_offset = id * elements_per_flit;

        if (global_data_offset >= 0) {
            int remaining_data = packet->message.data.size() - global_data_offset;
            current_payload_size = std::min(elements_per_flit, std::max(0, remaining_data));
        }
    }
}

float Flit::get_data(int local_index) const {
    if (global_data_offset + local_index < packet->message.data.size()) {
        return packet->message.data[global_data_offset + local_index];
    }
    return 0.0f;
}

int Flit::get_payload_size() const {
    return current_payload_size;
}

void Flit::update_data(int local_index, float new_val) {
    if (global_data_offset + local_index < packet->message.data.size()) {
        packet->message.data[global_data_offset + local_index] = new_val;
    }
}