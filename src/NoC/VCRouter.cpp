/*
 * VCRouter.cpp
 *
 */

#include "VCRouter.hpp"
#include "../parameters.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>

VCRouter::VCRouter(int* t_id, int in_out_port_num, VCNetwork* t_vcNetwork, int t_vn_num, int t_vc_per_vn, int t_vc_priority_per_vn, int t_in_depth)
{
    id[0] = t_id[0];  
    id[1] = t_id[1];  

    vcNetwork = t_vcNetwork;

    in_port_list.reserve(in_out_port_num);
    out_port_list.reserve(in_out_port_num);
    
    local_weights.clear();
    local_kv_cache.clear();

    mfu_occupied_until = 0;

    int total_vcs_per_port = t_vn_num * (t_vc_per_vn + t_vc_priority_per_vn);
    
    // [5 ports: up, right, down, left, local] x [N VCs per port]
    vc_compute_state.resize(in_out_port_num, std::vector<ComputeVCState>(total_vcs_per_port));

    for (int i=0; i< in_out_port_num; i++){
        Link * link = new Link((RInPort*)NULL);
        RInPort * t_rInPort = new RInPort(i, t_vn_num, t_vc_per_vn, t_vc_priority_per_vn, t_in_depth, this, link); 
        t_rInPort->rid[0] = t_id[0];
        t_rInPort->rid[1] = t_id[1];

        link->rInPort = t_rInPort;
        in_port_list.push_back(t_rInPort);

    #ifdef outPortNoInfinite
        ROutPort * t_rOutPort = new ROutPort(i, 1, 1, 0, 1); 
        if(i==4) 
        {
            t_rOutPort = new ROutPort(i, 1, 1, 0, 2);
        }
    #else
        ROutPort * t_rOutPort = new ROutPort(i, 1, 1, 0, INFINITE);
    #endif
        out_port_list.push_back(t_rOutPort);
    }
    rr_port = 0;
    rr_out_port = 0;
    port_num = in_out_port_num;
    port_total_utilization = 0;
    port_utilization_innet = 0;
    
    current_sram_usage = 0;
    local_weights.reserve(ROUTER_SRAM_LIMIT);
    local_kv_cache.reserve(ROUTER_SRAM_LIMIT);
    
    int total_vcs = t_vn_num * (t_vc_per_vn + t_vc_priority_per_vn);
    vc_compute_state.resize(in_out_port_num, std::vector<ComputeVCState>(total_vcs));

    local_kv_cache.reserve(ROUTER_SRAM_LIMIT);
    local_weights.reserve(ROUTER_SRAM_LIMIT);
    kv_token_count = 0;
}

int VCRouter::getRoute(Flit* t_flit){
      // cNoC: Source Routing (Wormhole Safe)
      if (t_flit->packet->message.type == 4 || t_flit->packet->message.type == 5) {
          int path_idx = t_flit->packet->current_path_index;
          auto& path = t_flit->packet->message.routing_path;

          if (path_idx < path.size() && path[path_idx] == (id[0] * X_NUM + id[1])) {
              path_idx++;
              t_flit->packet->current_path_index = path_idx;
          }
          
          if (path_idx < path.size()) {
              int next_router = path[path_idx];
              int target_x = next_router / X_NUM;
              int target_y = next_router % X_NUM;
              if (target_y < id[1]) return 3; // turn left
              if (target_y > id[1]) return 1; // turn right
              if (target_x < id[0]) return 0; // turn up
              if (target_x > id[0]) return 2; // turn down
          }
      }

      int x = t_flit->packet->destination[0];
      int y = t_flit->packet->destination[1];
      int z = t_flit->packet->destination[2];
      
      if(y < id[1]) return 3; 
      else if(y > id[1]) return 1; 
      else { 
          if(x < id[0]) return 0; 
          else if(x > id[0]) return 2; 
          else return (z+4); 
      }
      return -1;
}

void VCRouter::processDistributionPacket(Flit* t_flit) {
    int flat_router_id = id[0] * X_NUM + id[1]; 

    if (t_flit->computed_routers.test(flat_router_id)) {
        return;
    }
    
    t_flit->computed_routers.set(flat_router_id);

    int payload_size = t_flit->get_payload_size();
    int opcode = t_flit->packet->message.compute_op;

    if (opcode == ATTENTION) { // Opcode for KV Cache loading
        /*
         * STREAMING LLM IMPLEMENTATION (Attention Sinks + Rolling KV Cache) -> https://arxiv.org/abs/2309.17453
         * The paper demonstrates that a surprisingly large amount of attention score (often >50%) 
         * is allocated to the very first tokens ("Attention Sinks"). By keeping the KV cache of 
         * just the first 4 tokens fixed, and using a sliding window for the most recent tokens 
         * (Rolling KV Cache), a Transformer can generalize to infinite-length sequences without 
         * fine-tuning, keeping stability and perplexity unchanged.
         * * [HARDWARE MAPPING]
         * In our cNoC architecture, the "Rolling KV Cache" is physically implemented as a Ring Buffer 
         * inside the Router's SRAM. We lock the first 4 tokens (SINK_TOKENS) and circularly overwrite 
         * the remaining space up to the ROUTER_SRAM_LIMIT to prevent hardware memory overflow.
         * ====================================================================================
         */
        const int SINK_TOKENS = 4;
        
        // A KV loading packet carries [K, V] for a token, so the size is twice the head_dim
    //     int floats_per_token = payload_size; 
    //     int max_tokens_in_sram = ROUTER_SRAM_LIMIT / floats_per_token;

    //     if (local_kv_cache.size() + payload_size <= ROUTER_SRAM_LIMIT) {
    //         // PREFILL phase: There is still space, insert normally
    //         for (int i = 0; i < payload_size; i++) {
    //             local_kv_cache.push_back(t_flit->get_data(i));
    //         }
    //         current_sram_usage += payload_size;
    //     } 
    //     else {
    //         // SRAM FULL: Eviction is triggered (Ring Buffer)
    //         // The index rotates discarding intermediate tokens but SAVING the Sinks (0, 1, 2, 3)
    //         int ring_index = SINK_TOKENS + ((kv_token_count - SINK_TOKENS) % (max_tokens_in_sram - SINK_TOKENS));
    //         int offset = ring_index * floats_per_token;
            
    //         // Overwrite old data in O(1), zero memory reallocations
    //         for (int i = 0; i < payload_size; i++) {
    //             local_kv_cache[offset + i] = t_flit->get_data(i);
    //         }
    //     }
    //     kv_token_count++;
    // } 
        int floats_per_token = t_flit->packet->message.data_length; 
        int max_tokens_in_sram = ROUTER_SRAM_LIMIT / floats_per_token;

        if (kv_token_count < max_tokens_in_sram) {
            // Prefill phase
            int write_pos = (kv_token_count * floats_per_token) + t_flit->global_data_offset;
            
            for (int i = 0; i < payload_size; i++) {
                writeKV(write_pos + i, t_flit->get_data(i));
            }
        } 
        else {
            // Eviction phase (Ring Buffer)
            int ring_index;
            if (max_tokens_in_sram > SINK_TOKENS) {
                ring_index = SINK_TOKENS + ((kv_token_count - SINK_TOKENS) % (max_tokens_in_sram - SINK_TOKENS));
            } else {
                ring_index = std::max(0, max_tokens_in_sram - 1);
            }
            
            int base_offset = ring_index * floats_per_token;
            
            for (int i = 0; i < payload_size; i++) {
                // Secure overwrite (writeKV will understand that index < size and won't allocate more SRAM)
                writeKV(base_offset + t_flit->global_data_offset + i, t_flit->get_data(i));
            }
        }
        
        if (t_flit->type == 1 || t_flit->type == 10) {
            kv_token_count++;
        }
    } else {
        if (t_flit->id == 0) { 
             this->clearWeights();
        }
        
        int payload_size = t_flit->get_payload_size();
        for (int i = 0; i < payload_size; i++) {
            storeWeight(t_flit->get_data(i));
        }
    }
}

void VCRouter::computeInTransit(Flit* t_flit, int port_idx) {
    int this_router_id = id[0] * X_NUM + id[1];

    // Avoid redundant executions of the same flit if it passes multiple times
    // if (std::find(t_flit->computed_routers.begin(), t_flit->computed_routers.end(), this_router_id) != t_flit->computed_routers.end()) {
    //     return;
    // }
    // t_flit->computed_routers.push_back(this_router_id);

    // Avoid redundant executions of the same flit if it passes multiple times
    if (t_flit->computed_routers[this_router_id]) {
        return;
    }
    t_flit->computed_routers[this_router_id] = true;

    ComputeVCState& vc_state = vc_compute_state[port_idx][t_flit->vc];

    int opcode = -1;

    if (t_flit->type == 0 || t_flit->type == 10) { 
        vc_state.reset();
        vc_state.is_active = true;
        vc_state.compute_op = t_flit->packet->message.compute_op;
        
        if (!t_flit->packet->message.routing_path.empty() && 
            t_flit->packet->message.routing_path.front() == this_router_id) {
            
            int heads = (t_flit->packet->message.n_heads > 0) ? t_flit->packet->message.n_heads : 1;
            vc_state.running_max.assign(heads, -1e9);
            vc_state.running_sum.assign(heads, 0.0);
        } else {
            vc_state.running_max = t_flit->packet->message.running_max;
            vc_state.running_sum = t_flit->packet->message.running_sum;
        }
    }

    if (vc_state.is_active) {
        opcode = vc_state.compute_op;
    } else {
        return;
    }

    int payload_size = t_flit->get_payload_size();
    
    if (payload_size > 0) {
        switch(opcode) {
            case MATMUL:
                // MATMUL (15) and LINEAR (0) share the exact same mathematical logic 
                // (matrix-vector multiplication for the assigned tasks). 
                // Omitting the 'break' here allows MATMUL to cascade directly into 
                // the LINEAR block, avoiding unnecessary code duplication.
            case 0:                                         // LINEAR
            {
                int num_tasks = assigned_tasks.size();
                if (num_tasks == 0) break;
                
                int chunk_offset = t_flit->packet->message.chunk_offset;
                int chunk_row_size = t_flit->packet->message.chunk_row_size;
                
                for (int t = 0; t < num_tasks; t++) {
                    int task_id = assigned_tasks[t];
                    float local_accum = 0.0f;
                    
                    for (int i = 0; i < payload_size; ++i) {
                        int input_idx = t_flit->global_data_offset + i;
                        
                        if (input_idx >= t_flit->packet->message.psum_offset) continue; 
                        
                        if (input_idx >= chunk_offset && input_idx < chunk_offset + chunk_row_size) {
                            int w_offset = (t * chunk_row_size) + (input_idx - chunk_offset); 
                            if (w_offset < local_weights.size()) {
                                local_accum += t_flit->get_data(i) * local_weights[w_offset];
                            }
                        }
                    }
                    
                    int target_index = t_flit->packet->message.psum_offset + task_id;
                    if (target_index < t_flit->packet->message.data.size()) {
                        t_flit->packet->message.data[target_index] += local_accum;
                    }
                }
                break;
            }
            case ADD:
            {
                int num_tasks = assigned_tasks.size();
                for (int t = 0; t < num_tasks; t++) {
                    int task_id = assigned_tasks[t];
                    int required_flit_offset = task_id * 2;
                    
                    if (required_flit_offset >= t_flit->global_data_offset && 
                        required_flit_offset + 1 < t_flit->global_data_offset + payload_size) {
                        
                        int local_offset = required_flit_offset - t_flit->global_data_offset;
                        float x = t_flit->get_data(local_offset);
                        float res = t_flit->get_data(local_offset + 1);
                        
                        t_flit->packet->message.data[required_flit_offset] = x + res;
                    }
                }
                break;
            }
            case SWIGLU:
            {
                // int num_tasks = assigned_tasks.size();
                // for (int t = 0; t < num_tasks; t++) {
                //     int task_id = assigned_tasks[t];
                //     int required_flit_offset = task_id * 2;
                    
                //     if (required_flit_offset >= t_flit->global_data_offset && 
                //         required_flit_offset + 1 < t_flit->global_data_offset + payload_size) {
                        
                //         int local_offset = required_flit_offset - t_flit->global_data_offset;
                //         float gate = t_flit->get_data(local_offset);
                //         float up = t_flit->get_data(local_offset + 1);
                        
                //         float silu = gate * (1.0f / (1.0f + std::exp(-gate)));
                        
                //         t_flit->packet->message.data[required_flit_offset] = silu * up;
                //     }
                // }
                break;
            }
            case ATTENTION:
            {
                if (t_flit->type != 0 && t_flit->type != 10) break;
                if (local_kv_cache.empty()) break;

                int q_dim = t_flit->packet->message.data.size() - t_flit->packet->message.psum_offset;
                int k_dim = (t_flit->packet->message.k_dim > 0) ? t_flit->packet->message.k_dim : q_dim;
                int n_heads = t_flit->packet->message.n_heads;
                if (n_heads <= 0) n_heads = 1;
                
                int head_dim = q_dim / n_heads;
                int total_kv_heads = std::max(1, k_dim / head_dim);
                
                int N = t_flit->packet->message.routing_path.size() - 1;
                int router_idx = -1;
                for (int i = 0; i < N; i++) {
                    if (t_flit->packet->message.routing_path[i] == this_router_id) {
                        router_idx = i;
                        break;
                    }
                }
                
                if (router_idx == -1) break;

                int current_query_y = t_flit->packet->message.sequence_id;
                bool is_last_reduction_node = (router_idx == N - 1);

                const int SINK_TOKENS = 4;
                int floats_per_token = k_dim * 2;
                int max_tokens_in_sram = ROUTER_SRAM_LIMIT / floats_per_token;
                
                int current_local_count = 0;
                for (int y = 0; y <= current_query_y; y++) {
                    if (y % N == router_idx) current_local_count++;
                }

                std::vector<int> valid_physical_slots;
                int local_seq_id = 0;
                
                for (int y = 0; y <= current_query_y; y++) {
                    if (y % N == router_idx) {
                        bool retained = true;
                        
                        if (current_local_count > max_tokens_in_sram) {
                            int max_recent = max_tokens_in_sram - SINK_TOKENS;
                            if (local_seq_id >= SINK_TOKENS && local_seq_id < current_local_count - max_recent) {
                                retained = false; 
                            }
                        }

                        if (retained) {
                            int physical_slot;
                            if (local_seq_id < max_tokens_in_sram) {
                                physical_slot = local_seq_id;
                            } else {
                                if (max_tokens_in_sram > SINK_TOKENS) {
                                    physical_slot = SINK_TOKENS + ((local_seq_id - SINK_TOKENS) % (max_tokens_in_sram - SINK_TOKENS));
                                } else {
                                    physical_slot = std::max(0, max_tokens_in_sram - 1);
                                }
                            }
                            valid_physical_slots.push_back(physical_slot);
                        }
                        local_seq_id++;
                    }
                }
                
                int effective_tokens = valid_physical_slots.size();
                if (effective_tokens == 0) break;

                for (int h = 0; h < n_heads; h++) {
                    int kv_h = (total_kv_heads == 1) ? 0 : (h * total_kv_heads / n_heads); 
                    
                    std::vector<double> local_scores(effective_tokens, 0.0);
                    double local_max = -1e9;

                    for (int i = 0; i < effective_tokens; i++) {
                        int physical_slot = valid_physical_slots[i];
                        double dot_product = 0.0;
                        int k_offset = physical_slot * (k_dim * 2) + (kv_h * head_dim);
                        
                        for (int d = 0; d < head_dim; d++) {
                            dot_product += (double)t_flit->packet->message.data[(h * head_dim) + d] * (double)local_kv_cache[k_offset + d];
                        }

                        dot_product /= std::sqrt((double)head_dim);
                        local_scores[i] = dot_product;
                        if (dot_product > local_max) local_max = dot_product;
                    }
                    
                    double m_old = vc_state.running_max[h];
                    double l_old = vc_state.running_sum[h];
                    
                    double m_new = std::max(m_old, local_max);
                    double old_scale = std::exp(m_old - m_new);
                    double local_sum_exp = 0.0;
                    
                    for (int i = 0; i < effective_tokens; i++) {
                        local_scores[i] = std::exp(local_scores[i] - m_new);
                        local_sum_exp += local_scores[i];
                    }
                    double l_new = (l_old * old_scale) + local_sum_exp;
                    
                    for (int d = 0; d < head_dim; d++) {
                        int target_idx = t_flit->packet->message.psum_offset + (h * head_dim) + d;
                        
                        double current_o = (double)t_flit->packet->message.data[target_idx] * old_scale;
                        
                        double local_v_contribution = 0.0;
                        for (int i = 0; i < effective_tokens; i++) {
                            int physical_slot = valid_physical_slots[i];
                            int v_offset = physical_slot * (k_dim * 2) + k_dim + (kv_h * head_dim);
                            local_v_contribution += local_scores[i] * (double)local_kv_cache[v_offset + d];
                        }
                        
                        double final_o = current_o + local_v_contribution;
                        
                        if (is_last_reduction_node && l_new > 0.0) {
                            final_o /= l_new;
                        }
                        
                        t_flit->packet->message.data[target_idx] = (float)final_o;
                    }
                    
                    vc_state.running_max[h] = m_new;
                    vc_state.running_sum[h] = l_new;
                }
                
                t_flit->packet->message.running_max = vc_state.running_max;
                t_flit->packet->message.running_sum = vc_state.running_sum;
                
                break;
            }
        }
    }

    // Clean up the VC state table when the tail flit passes through
    if (t_flit->type == 1 || t_flit->type == 10) {
        vc_state.reset();
    }
}

void VCRouter::vcRequest(){  
  for(int i=0; i<port_num; i++){ 
      in_port_list[(i+rr_port)%port_num]->vc_request();
  }
}

void VCRouter::getSwitch(){
  for(int i=0; i<port_num; i++){ 
      in_port_list[(i+rr_port)%port_num]->getSwitch();
  }
  rr_port = (rr_port+1)%port_num;
}

void VCRouter::outPortDequeue(){
    for(int count = 0; count < port_num; count++){ 
        int i = (rr_out_port + count) % port_num;

        if(out_port_list[i]->buffer_list[0]->cur_flit_num != 0 && out_port_list[i]->buffer_list[0]->read()->sched_time < cycles){
            Flit* flit = out_port_list[i]->buffer_list[0]->read();

            if (flit->packet->message.type == 4 || flit->packet->message.type == 5) {
                if (cycles < mfu_occupied_until) {
                    continue;
                }
            }
            
            if (flit->packet->message.type == 5 && flit->packet->message.compute_op == ATTENTION) {
                int this_router_id = id[0] * X_NUM + id[1];
                int N = flit->packet->message.routing_path.size() - 1;
                int router_idx = -1;
                for (int r = 0; r < N; r++) {
                    if (flit->packet->message.routing_path[r] == this_router_id) {
                        router_idx = r; break;
                    }
                }
                if (router_idx != -1) {
                    int current_query_y = flit->packet->message.sequence_id;
                    int expected_local_tokens = 0;
                    for (int y = 0; y <= current_query_y; y++) {
                        if ((y % N) == router_idx) expected_local_tokens++;
                    }
                    
                    if (kv_token_count < expected_local_tokens) {
                        mfu_occupied_until = cycles + 2;
                        continue;
                    }
                }
            }

            flit = out_port_list[i]->buffer_list[0]->dequeue();

            int compute_delay = 0;

            if (flit->packet->message.type == 4) {
                int this_router = id[0] * X_NUM + id[1];
                
                // Snooping: The router intercepts the in-transit data and populates the SRAM
                bool is_target = false;
                for (int r : flit->packet->message.routing_path) {
                    if (r == this_router) is_target = true;
                }
                if (flit->packet->destination[0] == id[0] && flit->packet->destination[1] == id[1]) {
                    is_target = true;
                }

                if (is_target) {
                    processDistributionPacket(flit);
                }
                
                mfu_occupied_until = cycles + 1; 
                compute_delay = 1;
            }
            else if (flit->packet->message.type == 5) {
                computeInTransit(flit, i); 
                
                int opcode = flit->packet->message.compute_op;
                
                if (opcode == SWIGLU || opcode == GEGLU) {
                    compute_delay = 1; 
                } 
                else if (opcode == ADD) {
                    compute_delay = ADD_LATENCY;
                }
                else if (opcode == ATTENTION) {
                    // Softmax and projections are only completed when the tail flit arrives.
                    if (flit->type == 1 || flit->type == 10) {
                        
                        // Calculate the number of local tokens currently in the router's cache
                        int q_dim = flit->packet->message.data.size() - flit->packet->message.psum_offset;
                        int num_local_tokens = 0;
                        if (q_dim > 0) {
                            num_local_tokens = local_kv_cache.size() / (q_dim * 2);
                        }
                        
                        // Obtain the hardware cycles required for the operations, using PE_NUM_OP as an 
                        // indicator of the internal parallelization of the router's Special Function Unit (SFU).
                        int dot_product_ops = (num_local_tokens * q_dim) / PE_NUM_OP + 1; // Q * K^T
                        int v_proj_ops      = (num_local_tokens * q_dim) / PE_NUM_OP + 1; // Softmax * V
                        int softmax_mac_ops = (3 * num_local_tokens) / PE_NUM_OP + 1;     // Linear operations of Softmax
                        
                        // Latency = (Linear Ops * MAC Cycles) + (Non-Linear Ops * Specific Cycles)
                        compute_delay = (dot_product_ops + v_proj_ops + softmax_mac_ops) * MAC_LATENCY 
                                        + (num_local_tokens * EXP_LATENCY) 
                                        + DIV_LATENCY 
                                        + SQRT_LATENCY;
                                        
                    } else {
                        // Intermediate flits (Head, Body) pass through the router in a pipelined 
                        // fashion while the SFU is accumulating data, so they only cost 1 cycle.
                        compute_delay = 1; 
                    }
                } else {
                    compute_delay = MAC_LATENCY; 
                }

                mfu_occupied_until = cycles + compute_delay; 
            }

            out_port_list[i]->out_link->rInPort->buffer_list[flit->vc]->enqueue(flit);
            
            flit->sched_time = cycles + compute_delay + LINK_TIME - 1;

            flit->trace_node.push_back(out_port_list[i]->out_link->rInPort->rid[0]*X_NUM + out_port_list[i]->out_link->rInPort->rid[1]);
            flit->trace_time.push_back(cycles + LINK_TIME - 1);

            port_total_utilization++;
            if (i<=3) port_utilization_innet++;

            if(flit->type == 0 || flit->type == 10){
                VCRouter* vcRouter = dynamic_cast<VCRouter*>(out_port_list[i]->out_link->rInPort->router_owner);
                if (vcRouter != NULL){
#ifdef SHARED_VC 
                    if(flit->packet->message.QoS == 1){
                        out_port_list[i]->out_link->rInPort->priority_vc.push_back(flit->vc);
                        out_port_list[i]->out_link->rInPort->priority_switch.push_back(flit->vc);
                    }
#endif
                    int route_result = vcRouter->getRoute(flit);
                    out_port_list[i]->out_link->rInPort->out_port[flit->vc] = route_result;
                    assert(out_port_list[i]->out_link->rInPort->state[flit->vc] == 1);
                }
                out_port_list[i]->out_link->rInPort->state[flit->vc] = 2; 
            }
        }
    }
  rr_out_port = (rr_out_port + 1) % port_num;
}

void VCRouter::runOneStep(){
    vcRequest();
    getSwitch();
    outPortDequeue();
}

bool VCRouter::allocateSRAM(int num_floats) {
    if (current_sram_usage + num_floats > ROUTER_SRAM_LIMIT) {
        std::cerr << "\n[HARDWARE EXCEPTION] Router (" << id[0] << "," << id[1] 
                  << ") SRAM Overflow! Limit: " << ROUTER_SRAM_LIMIT 
                  << " Used: " << current_sram_usage 
                  << " Requested: " << num_floats << std::endl;
        return false; 
    }
    current_sram_usage += num_floats;
    return true;
}

void VCRouter::storeWeight(float weight_value) {
    bool can_allocate = allocateSRAM(1);
    if (!can_allocate) {
        std::cerr << "FATAL ERROR: SRAM OVERFLOW! Impossible to store weight in router (" 
                  << id[0] << "," << id[1] << ")." << std::endl;
        exit(EXIT_FAILURE);
    }
    local_weights.push_back(weight_value);
}

void VCRouter::storeKV(float kv_value) {
    bool can_allocate = allocateSRAM(1);
    if (!can_allocate) {
        std::cerr << "FATAL ERROR: SRAM OVERFLOW! Impossible to store KV cache in router (" 
                  << id[0] << "," << id[1] << ")." << std::endl;
        exit(EXIT_FAILURE);
    }
    local_kv_cache.push_back(kv_value);
}

void VCRouter::writeKV(int index, float kv_value) {
    if (index >= local_kv_cache.size()) {
        int needed_expansion = (index + 1) - local_kv_cache.size();
        
        bool can_allocate = allocateSRAM(needed_expansion);
        if (!can_allocate) {
            std::cerr << "FATAL ERROR: SRAM OVERFLOW! Impossible to expand KV cache in router (" 
                      << id[0] << "," << id[1] << ")." << std::endl;
            exit(EXIT_FAILURE);
        }
        
        local_kv_cache.resize(index + 1, 0.0f);
    }
    local_kv_cache[index] = kv_value;
}

void VCRouter::clearSRAM() {
    local_weights.clear();
    local_kv_cache.clear();
    current_sram_usage = 0;
    kv_token_count = 0;

    kv_token_count = 0;
    assigned_tasks.clear();
}

void VCRouter::clearWeights() {
    int weight_size = local_weights.size();
    local_weights.clear();
    current_sram_usage -= weight_size; 
}

VCRouter::~VCRouter ()
{
  RInPort* inPort;
    while(in_port_list.size()!=0){
        inPort = in_port_list.back();
        in_port_list.pop_back();
        delete inPort;
    }

    ROutPort* outPort;
        while(out_port_list.size()!=0){
            outPort = out_port_list.back();
            out_port_list.pop_back();
            delete outPort;
    }
}