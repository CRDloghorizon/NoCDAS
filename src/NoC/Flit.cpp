/*
 * flit.cpp
 *
 */

#include "Flit.hpp"
#include "../parameters.hpp"

const int Flit::length = FLIT_LENGTH;
std::vector<Flit*> Flit::free_pool;

Flit::Flit(int t_id, int t_type, int t_vnet, int t_vc, Packet* t_packet, float t_cycles, int t_pid){
    id = t_id;
    type = t_type;
    vnet = t_vnet;
    vc = t_vc;
    out_port = -1;
    packet = t_packet;
    sched_time = t_cycles;
    packetid=t_pid;
    
    trace_node.reserve(X_NUM + Y_NUM); 
    trace_time.reserve(X_NUM + Y_NUM);

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

Flit* Flit::allocate(int t_id, int t_type, int t_vnet, int t_vc, Packet* t_packet, float t_cycles, int t_pid) {
    if (!free_pool.empty()) {
        Flit* flit = free_pool.back();
        free_pool.pop_back();
        flit->reset(t_id, t_type, t_vnet, t_vc, t_packet, t_cycles, t_pid);
        return flit;
    }
    return new Flit(t_id, t_type, t_vnet, t_vc, t_packet, t_cycles, t_pid);
}

void Flit::release(Flit* flit) {
    free_pool.push_back(flit);
}

void Flit::reset(int t_id, int t_type, int t_vnet, int t_vc, Packet* t_packet, float t_cycles, int t_pid) {
    id = t_id;
    type = t_type;
    vnet = t_vnet;
    vc = t_vc;
    out_port = -1;
    packet = t_packet;
    sched_time = t_cycles;
    packetid = t_pid;

    trace_node.clear();
    trace_time.clear();

    computed_routers.reset(); 

    current_payload_size = 0;
    global_data_offset = 0;

    if (packet != nullptr && (packet->message.type == 4 || packet->message.type == 5)) {
        int elements_per_flit = FLIT_LENGTH / DATA_BYTES;
        global_data_offset = id * elements_per_flit;

        if (global_data_offset >= 0) {
            int remaining_data = packet->message.data.size() - global_data_offset;
            current_payload_size = std::min(elements_per_flit, std::max(0, remaining_data));
        }
    }
}