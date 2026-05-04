/*
 * MAC.cpp
 *
 */

#include "MAC.hpp"

MAC::MAC (int t_id, MACnet* t_net, int t_NI_id)
{
    id = t_id;
    net = t_net;
    NI_id = t_NI_id;
    weight.clear();
    infeature.clear();
    inbuffer.clear();
    ch_size = 0;
    m_size = 0;
    fn = -1;
    tmpch = -1;
    tmpm = 0;
    request = -1;
    tmp_request = -1;
    cached_score_row = -1;
    cached_score_head = -1;

    outfeature = 0.0;
    nextMAC = NULL;
    pecycle = 0;
    selfstatus = 0;
    send = 0;
    m_count = 0;

    // for new pooling
    npoolflag = 0;
    n_tmpch = 0;
    n_tmpm.clear();

    // for Transformer
    causal_mask = 0;
    local_sram_usage = 0;

    current_chunk = 0;
    total_chunks = 1;
    psum_accumulator = 0.0f;

    // find dest id
    int xid = NI_id / X_NUM;
    int yid = NI_id % X_NUM;
    // MC nodes
#ifdef MemNode4
    if (xid <= 3)
    {
        dest_mem_id = dest_list[(yid/4)];
    }
    else
    {
        dest_mem_id = dest_list[(yid/4) + 2];
    }
#elif defined MemNode2
    dest_mem_id = dest_list[(yid/2)];
#elif defined MemNode8
    if (xid <= 3) {
        dest_mem_id = dest_list[(yid/2)];
    } else {
        dest_mem_id = dest_list[(yid/2) + 4];
    }
#elif defined MemNode8edge
    if (yid <= 3) {
        dest_mem_id = dest_list[(xid/2)*2];     // left 0 2 4 6
    } else {
        dest_mem_id = dest_list[(xid/2)*2 + 1]; // right 1 3 5 7
    }
#elif defined MemNode18                         // 12*12
    if (xid <= 3) {
        dest_mem_id = dest_list[(yid/2)];
    } else if (xid <= 7 && xid > 3 ) {
        dest_mem_id = dest_list[(yid/2) + 6];
    } else {
        dest_mem_id = dest_list[(yid/2) + 12];
    }
#elif defined MemNode32                         // 16*16
    if (xid <= 3) {
        dest_mem_id = dest_list[(yid/2)];
    } else if (xid <= 7 && xid > 3 ) {
        dest_mem_id = dest_list[(yid/2) + 8];
    } else if (xid <= 11 && xid > 7 ) {
        dest_mem_id = dest_list[(yid/2) + 16];
    } else {
        dest_mem_id = dest_list[(yid/2) + 24];
    }
#elif defined MemNode5                          // 6*6
    if (xid <= 3) {
        dest_mem_id = dest_list[(yid/2)];
    } else {
        dest_mem_id = dest_list[(yid/3) + 3];
    }
#elif defined MemNode13                         // 10*10
    if (xid <= 3) {
        dest_mem_id = dest_list[(yid/2)];
    } else if (xid <= 7 && xid > 3 ) {
        dest_mem_id = dest_list[(yid/2) + 5];
    } else {
        dest_mem_id = dest_list[(yid/4) + 10];
    }
#endif
    routing_table.clear();
}


bool MAC::inject (int type, int d_id, int data_length, float t_output, NI* t_NI, int p_id, int mac_src)
{
    Message msg;
    msg.NI_id = NI_id;
    msg.mac_id = mac_src;                       //MAC
    msg.data_length = data_length;
    int selector = rand()%90;
#ifdef LCS_URS_TRAFFIC
    if(selector >= 45) 
        msg.QoS = 3;
    else 
        msg.QoS = 0;
#endif
#ifdef SHARED_VC
    if(msg.QoS == 3) 
        msg.QoS = 1;
#endif
    msg.QoS = 0;

    msg.data.assign(1, t_output);
    msg.data.push_back(tmpch);
    msg.data.push_back(tmpm);

    msg.penable = this->npoolflag;

    msg.destination = d_id;
    msg.out_cycle = pecycle;
    msg.sequence_id = 0;
    msg.signal_id = p_id;
    msg.slave_id = d_id;                       //NI
    msg.source_id = NI_id;                     // NI
    msg.type = type;                           // 0 1 2 3

    Packet* packet = new Packet(std::move(msg), X_NUM, t_NI->NI_num);
    packet->send_out_time = pecycle;
    packet->in_net_time = pecycle;
    net->vcNetwork->NI_list[NI_id]->packetBuffer_list[packet->vnet]->enqueue(packet);

    return true;
}


void MAC::runOneStep()
{
    // output stationary (neuron based calculation)
    if (pecycle < cycles){
        // initial idle state
        int stats1;
        if(selfstatus == 0)
        {
            if(routing_table.size()==0)
            {
                selfstatus = 0;
                pecycle = cycles;
            }
            else
            {
                pecycle = cycles;
                selfstatus = 1;
            }
        }
        // request data state
        else if(selfstatus == 1)
        {
            if (current_chunk == 0) {
                request = routing_table.front();
                tmp_request = request;
                routing_table.pop_front();
            }
            inject(0, dest_mem_id, 1, request, net->vcNetwork->NI_list[NI_id], packet_id + request, id);
            selfstatus = 2;
            pecycle = cycles;
#ifdef Countlatency
            stats1 = (packet_id + tmp_request)*3;
            if(stats1 < CountNum) {
                DNN_latency[stats1][0] = net->c_layer;
                DNN_latency[stats1][1] = 0;
                DNN_latency[stats1][2] = id;
                DNN_latency[stats1][3] = pecycle;
            }
#endif
        }
        else if(selfstatus == 2)
        {
            if(request >= 0) {pecycle = cycles; selfstatus = 2; return;}
            assert((inbuffer.size() >= 4) && "Inbuffer not correct after request is set to 0");

            // inbuffer: [fn]
            fn = inbuffer[0]; 
#ifdef newpooling
            if(this->npoolflag == 1 && this->n_tmpch == -1)
            {
                assert((fn == 10) && "Inbuffer not correct when merged with pooling");
                if(this->routing_table.size()==0)
                {
                    this->selfstatus = 5;
                    this->send = 3;
                }
                else
                {
                    this->selfstatus = 0;               // back to initial state
                    this->send = 0;
                }
                // cout << "from mac " << this->id << " abandoned at cycles " << cycles << " " << selfstatus << endl;
                this->weight.clear();
                this->infeature.clear();
                this->inbuffer.clear();
                this->outfeature = 0.0;
                this->npoolflag = 0;
                this->n_tmpch = -1;
                this->n_tmpm.clear();
                this->pecycle = cycles + 1; //cycles + 1
                return;
            }
#endif
            if (fn >=0 && fn <=3){                     // Conv [fn] [ch size] [map size] [i] [w + b]
                ch_size = inbuffer[1];
                m_size = inbuffer[2];
                infeature.assign(inbuffer.begin() + 3, inbuffer.begin() + 3 + ch_size * m_size); //input
                weight.assign(inbuffer.begin() + 3 + ch_size * m_size, inbuffer.end()); // w matrix + b (ch_size * m_size + 1)
                assert((weight.size() == ch_size * m_size + 1) && "Weight not correct after request (Conv)");
            }
            else if (fn >= 4 && fn <= 7)              // fc [fn] [map size] [i] [w + b]
            {
                ch_size = 1;
                m_size = inbuffer[1];
                infeature.assign(inbuffer.begin() + 2, inbuffer.begin() + 2 + m_size);
                weight.assign(inbuffer.begin() + 2 + m_size, inbuffer.end()); //w + b
            }
            else if (fn == 8 || fn == 12)           // max or avg pooling [fn] [map size] [i]
            {
                ch_size = 1;
                m_size = inbuffer[1];
                infeature.assign(inbuffer.begin() + 2, inbuffer.end());
                assert((infeature.size() == m_size) && "Inbuffer not correct after request (pooling)");
            }
            else if (fn >= MATMUL && fn <= ATTENTION) // Layer Transformer
            {
                ch_size = 1;
                if (fn == MATMUL || fn == ADD || fn == SWIGLU) { 
                    m_size = inbuffer[1];
                    if (fn == SWIGLU) {
                        // SwiGLU: the input size is 2 * m_size (Gate + Up), no weights
                        infeature.assign(inbuffer.begin() + 2, inbuffer.begin() + 4);
                    } else {
                        // MatMul, LayerNorm, Add, RMSNorm
                        infeature.assign(inbuffer.begin() + 2, inbuffer.begin() + 2 + m_size);
                        weight.assign(inbuffer.begin() + 2 + m_size, inbuffer.end());
                    }
                } else if (fn == LAYERNORM || fn == RMSNORM) {
                    m_size = 1;
                    infeature.assign(inbuffer.begin() + 2, inbuffer.end()); 
                } else if (fn == SOFTMAX_TR) {                                      // Softmax
                    m_size = inbuffer[1];
                    causal_mask = inbuffer[2];
                    infeature.assign(inbuffer.begin() + 3, inbuffer.end());
                } else if (fn == EMBEDDING) {                                       // Embedding
                    m_size = inbuffer[1];
                    weight.assign(inbuffer.begin() + 2, inbuffer.end());
                } else if (fn == ROPE) {
                    infeature.assign(inbuffer.begin() + 4, inbuffer.begin() + 6);
                } else if (fn == ATTENTION) {                                       // Attention Hardware (Fused)
                    m_size = inbuffer[1];
                    infeature.assign(inbuffer.begin() + 7, inbuffer.end()); 
                }
            }

            outfeature = 0.0;
            selfstatus = 3;
            pecycle = cycles;
            return;
        }
        else if(selfstatus == 3){
            
#ifdef cNoC_MODE
            // In cNoC mode, the MAC units act like co-processors for global reductions.
            // They should not execute MatMul, Add, or SwiGLU operations, as those are performed in-transit on the routers.
            if (fn == MATMUL || fn == ADD || fn == SWIGLU) {
                cout << "FATAL ERROR: The MAC node " << id << " does not support executing function " 
                     << fn << " in cNoC_MODE! This operation is performed in-transit on the routers." << endl;
                outfeature = 0.0;
                selfstatus = 4;
                pecycle = cycles + 1;
                inject(2, dest_mem_id, 1, outfeature, net->vcNetwork->NI_list[NI_id], packet_id + tmp_request, id);
                return;
            }
#endif
            // normal MAC op
            if (fn >=0 && fn <=3) {                         // Conv
                float temp_sum = psum_accumulator; 
                
                for(int i=0; i < ch_size; i++) {
                    temp_sum += infeature[i] * weight[i];
                }
                
                if (current_chunk == total_chunks - 1) {
                    temp_sum += weight.back();
                }

                outfeature = temp_sum;
                psum_accumulator = temp_sum;
            }
            else if (fn >= 4 && fn <= 7)                    // FC
            {
                for(int j=0; j < m_size; j++) { outfeature += infeature[j] * weight[j]; }
                outfeature += weight[m_size];
            }
            else if (fn == 8)                               // max pooling
            {
                outfeature = infeature[0];
                for(int j=1; j < m_size; j++)
                {
                    if (infeature[j] > outfeature) {outfeature = infeature[j];}
                }
                selfstatus = 4;                             // ready for this computation
                pecycle = cycles + 1;                       // sync cycles

                inject(2, dest_mem_id, 1, outfeature, net->vcNetwork->NI_list[NI_id], packet_id + tmp_request, id);
#ifdef Countlatency
                stats1 = (packet_id + tmp_request)*3 + 2;
                if(stats1 < CountNum) {
                    DNN_latency[stats1][3] = pecycle;
                }
#endif
                return;
            }
            else if (fn == 12)                              // average pooling
            {
                outfeature = infeature[0];
                for(int j=1; j < m_size; j++)
                {
                    outfeature = outfeature + infeature[j];
                }
                outfeature = outfeature / m_size;
                selfstatus = 4;                             // ready for this computation
                pecycle = cycles + 1;                       // sync cycles

                inject(2, dest_mem_id, 1, outfeature, net->vcNetwork->NI_list[NI_id], packet_id + tmp_request, id);
#ifdef Countlatency
                stats1 = (packet_id + tmp_request)*3 + 2;
                if(stats1 < CountNum) {
                    DNN_latency[stats1][3] = pecycle;
                }
#endif
                //packet_id++;
                return;
            }
            else if (fn >= MATMUL && fn <= ATTENTION)                                               // Operazioni Transformer
            {
                if (fn == MATMUL) {                                                                 // MatMul
                    for(int j=0; j < m_size; j++) { outfeature += infeature[j] * weight[j]; }
                } 
                else if (fn == LAYERNORM) {                                                         // LayerNorm
                    // infeature = [mean, var, x_i, gamma, beta]
                    float mean  = infeature[0];
                    float var   = infeature[1];
                    float x_i   = infeature[2];
                    float gamma = infeature[3];
                    float beta  = infeature[4];
                    outfeature = (x_i - mean) / std::sqrt(var + 1e-5) * gamma + beta;
                }
                else if (fn == SOFTMAX_TR) {                                                        // Softmax
                    assert(m_size > 0 && "SOFTMAX_TR: m_size cannot be 0, check inbuffer after request");
                    assert((int)infeature.size() >= m_size && "SOFTMAX_TR: infeature too short for m_size");

                    int target_idx = tmpm % m_size;
                    int seq_idx = tmpm / m_size;
                    if (causal_mask == 1 && target_idx > seq_idx) { 
                        outfeature = 0.0; // causal masking
                    } else {
                        float max_val = -1e9;
                        int limit = (causal_mask == 1) ? seq_idx : m_size - 1;
                        for(int j=0; j <= limit; j++) { if (infeature[j] > max_val) max_val = infeature[j]; }
                        float sum_exp = 0.0;
                        for(int j=0; j <= limit; j++) { sum_exp += std::exp(infeature[j] - max_val); }
                        outfeature = std::exp(infeature[target_idx] - max_val) / sum_exp;
                    }
                } 
                else if (fn == ADD) {                                                               // Add (Residual Connection)
                    int idx = tmpm % m_size;
                    outfeature = infeature[idx] + weight[idx];
                } 
                else if (fn == EMBEDDING) {                                                         // Embedding Lookup
                    int idx = tmpm % m_size;
                    outfeature = weight[idx];
                }
                else if (fn == RMSNORM) {                                                           // RMSNorm
                    // infeature contains [rms, x_i, gamma]
                    float rms   = infeature[0];
                    float x_i   = infeature[1];
                    float gamma = infeature[2];
                    outfeature = (x_i / rms) * gamma;
                }
                else if (fn == SWIGLU) {                                                            // SwiGLU
                    float gate = infeature[0];
                    float up   = infeature[1]; 
                    
                    float silu = gate * (1.0 / (1.0 + std::exp(-gate)));
                    outfeature = silu * up;
                }
                else if (fn == ROPE) {                                                              // RoPE
                    int pos = inbuffer[2];       
                    int head_dim = inbuffer[3];  
                    int idx = tmpm % m_size;
                    
                    int d = idx % head_dim; 
                    int half_dim = head_dim / 2;
                    int feat_idx = (d < half_dim) ? d : (d - half_dim);
                    
                    float freq = 1.0 / std::pow(10000.0, ((float)(feat_idx * 2) / head_dim));
                    float theta = pos * freq;
                    float cos_val = std::cos(theta);
                    float sin_val = std::sin(theta);

                    float val_curr = infeature[0];
                    float val_pair = infeature[1];

                    if (d < half_dim) {
                        outfeature = val_curr * cos_val - val_pair * sin_val;
                    } else {
                        outfeature = val_curr * cos_val + val_pair * sin_val;
                    }
                }
                else if (fn == ATTENTION)                                                           // Attention (Hardware Fused Attention Layer with RoPE and Score Caching)
                {
                    int fused_dim = inbuffer[1];
                    int q_dim = inbuffer[2];
                    int k_dim = inbuffer[3];
                    int n_heads = inbuffer[4];
                    int current_row = inbuffer[5];
                    int target_idx = inbuffer[6];
                    
                    int head_dim = q_dim / n_heads;
                    int k_head_dim = k_dim / n_heads; 
                    int my_head = target_idx / head_dim;
                    int target_d = target_idx % head_dim;
                    
                    int n_tokens_in_buffer = infeature.size() / fused_dim;
                    #if ENABLE_KV_CACHE
                        bool is_cache_hit = (n_tokens_in_buffer == 1 && current_row > 0);
                    #else
                        bool is_cache_hit = false;
                    #endif

                    int q_offset = 0;
                    int k_offset = q_dim;
                    int v_offset = q_dim + k_dim;
                    int kv_size = k_dim * 2;
                    int total_kv_heads = k_dim / k_head_dim;
                    int k_half_dim = k_head_dim / 2;
                    int half_dim = head_dim / 2;

                    int kv_size_per_token = k_dim * 2;
                    int expected_cache_size = (current_row + 1) * kv_size_per_token;

                    // We put the data in the cache ONLY if the cache doesn't already have the data for this row.
                    if (this->kv_cache.size() < expected_cache_size) {
                        
                        if (!is_cache_hit || this->kv_cache.size() == 0) {
                            // Cache miss or prefill: Reconstruct the local cache by reading the entire history from the NoC.
                            this->kv_cache.clear();
                            for (int t = 0; t <= current_row; t++) {
                                int t_offset = t * fused_dim;
                                std::vector<float> token_k_rotated(k_dim, 0.0);

                                // RoPE
                                for (int h = 0; h < total_kv_heads; h++) {
                                    for (int d = 0; d < k_half_dim; d++) {
                                        float freq = 1.0 / std::pow(10000.0, (float)(2 * d) / k_head_dim);
                                        float theta = t * freq;
                                        float cos_val = std::cos(theta);
                                        float sin_val = std::sin(theta);

                                        float k1 = infeature[t_offset + k_offset + (h * k_head_dim) + d];
                                        float k2 = infeature[t_offset + k_offset + (h * k_head_dim) + d + k_half_dim];

                                        token_k_rotated[(h * k_head_dim) + d]              = k1 * cos_val - k2 * sin_val;
                                        token_k_rotated[(h * k_head_dim) + d + k_half_dim] = k2 * cos_val + k1 * sin_val;
                                    }
                                }
                                // Insert of rotated K and original V into the cache
                                this->kv_cache.insert(this->kv_cache.end(), token_k_rotated.begin(), token_k_rotated.end());
                                this->kv_cache.insert(this->kv_cache.end(), infeature.begin() + t_offset + v_offset, infeature.begin() + t_offset + v_offset + k_dim);
                            }
                        } else {
                            // Hit: infeature has only the current token, rotate it and append to the cache.
                            std::vector<float> token_k_rotated(k_dim, 0.0);
                            for (int h = 0; h < total_kv_heads; h++) {
                                for (int d = 0; d < k_half_dim; d++) {
                                    float freq = 1.0 / std::pow(10000.0, (float)(2 * d) / k_head_dim);
                                    float theta = current_row * freq;
                                    float cos_val = std::cos(theta);
                                    float sin_val = std::sin(theta);

                                    float k1 = infeature[k_offset + (h * k_head_dim) + d];
                                    float k2 = infeature[k_offset + (h * k_head_dim) + d + k_half_dim];

                                    token_k_rotated[(h * k_head_dim) + d]              = k1 * cos_val - k2 * sin_val;
                                    token_k_rotated[(h * k_head_dim) + d + k_half_dim] = k2 * cos_val + k1 * sin_val;
                                }
                            }
                            this->kv_cache.insert(this->kv_cache.end(), token_k_rotated.begin(), token_k_rotated.end());
                            this->kv_cache.insert(this->kv_cache.end(), infeature.begin() + v_offset, infeature.begin() + v_offset + k_dim);
                        }
                    }

                    // calculating score (O(N)) or cache recovery (O(1)) ---
                    int window_start = 0;
                    if (current_row >= MAX_CONTEXT_WINDOW) {
                        window_start = current_row - MAX_CONTEXT_WINDOW + 1;
                    }
                    
                    // Number of token taken into account (Max: MAX_CONTEXT_WINDOW)
                    int effective_history = current_row - window_start + 1;
                    int calctime = 0;
                    int kv_head_id = my_head * (k_dim / q_dim);

                    // Checking the local cache (based on the window, not the entire history)
                    bool scores_hit = (this->cached_score_row == current_row && this->cached_score_head == my_head);
                    std::vector<float> final_scores(effective_history, 0.0);

                    if (!scores_hit || this->cached_attention_scores.size() != effective_history) {                        
                        // RoPE for the Query
                        std::vector<float> q_rotated(head_dim, 0.0);
                        int current_token_offset = is_cache_hit ? 0 : (current_row * fused_dim);

                        for (int d = 0; d < half_dim; d++) {
                            float freq = 1.0 / std::pow(10000.0, (float)(2 * d) / head_dim);
                            float theta = current_row * freq;
                            float cos_val = std::cos(theta);
                            float sin_val = std::sin(theta);

                            float q1 = infeature[current_token_offset + q_offset + (my_head * head_dim) + d];
                            float q2 = infeature[current_token_offset + q_offset + (my_head * head_dim) + d + half_dim];

                            q_rotated[d]            = q1 * cos_val - q2 * sin_val;
                            q_rotated[d + half_dim] = q2 * cos_val + q1 * sin_val;
                        }

                        // Dot Product (Q * K^T) for the current window
                        float max_score = -1e9;

                        for (int t = window_start; t <= current_row; t++) { 
                            float dot_product = 0.0;
                            int k_start = (t * kv_size) + (kv_head_id * k_head_dim);
                            
                            for (int d = 0; d < k_head_dim; d++) {
                                dot_product += q_rotated[d] * this->kv_cache[k_start + d];
                            }
                            
                            int score_idx = t - window_start; 
                            final_scores[score_idx] = dot_product / std::sqrt((float)head_dim);
                            
                            if (final_scores[score_idx] > max_score) { 
                                max_score = final_scores[score_idx]; 
                            }
                        }
                        
                        // Softmax
                        float sum_exp = 0.0;
                        for (int i = 0; i < effective_history; i++) {
                            final_scores[i] = std::exp(final_scores[i] - max_score);
                            sum_exp += final_scores[i];
                        }
                        for (int i = 0; i < effective_history; i++) {
                            final_scores[i] /= sum_exp;
                        }

                        // cache score update
                        this->cached_attention_scores = final_scores;
                        this->cached_score_row = current_row;
                        this->cached_score_head = my_head;

                        // hardware timing
                        int dot_product_ops = (effective_history * k_head_dim) / PE_NUM_OP + 1;
                        int softmax_ops     = (3 * effective_history)          / PE_NUM_OP + 1;
                        int value_ops       = effective_history                / PE_NUM_OP + 1;
                        calctime = (dot_product_ops + softmax_ops + value_ops) * MAC_LATENCY + (effective_history * EXP_LATENCY) + DIV_LATENCY + CORDIC_LATENCY;

                    } else {
                        // cache hit, reuse the cache scores
                        final_scores = this->cached_attention_scores;
                        
                        // hardware timing
                        int value_ops = effective_history / PE_NUM_OP + 1;
                        calctime = value_ops * MAC_LATENCY;
                    }

                    // value projection
                    outfeature = 0.0; 
                    for (int t = window_start; t <= current_row; t++) {
                        int v_start = (t * kv_size) + k_dim + (kv_head_id * k_head_dim); 
                        int score_idx = t - window_start;
                        outfeature += final_scores[score_idx] * this->kv_cache[v_start + target_d];
                    }
                    
                    // NoC injection
                    calctime = calctime * PE_FREQ_RATIO;
                    selfstatus = 4;
                    pecycle    = cycles + calctime;

                    this->tmpm = (current_row * q_dim) + target_idx; 
                    inject(2, dest_mem_id, 1, outfeature, net->vcNetwork->NI_list[NI_id], packet_id + tmp_request, id);

                    #ifdef Countlatency
                        int stats1 = (packet_id + tmp_request) * 3 + 2;
                        if (stats1 < CountNum) {
                            DNN_latency[stats1][3] = pecycle;
                        }
                    #endif

                    return;
                }

                int calctime = 0;

                // dynamic calculation of the pipeline depth of the node
                // based on the global variable PE_NUM_OP 
                int SYSTOLIC_DIM = std::ceil(std::sqrt(PE_NUM_OP)); 

                if (fn == MATMUL) { 
                    // MatMul: overhead -> systolic array initialization + active cycles
                    int active_cycles = (m_size / PE_NUM_OP) + 1;
                    calctime = (SYSTOLIC_DIM + active_cycles) * MAC_LATENCY;
                } 
                else if (fn == LAYERNORM) { 
                    // LayerNorm (Mean, Variance, Norm)
                    int vector_ops = (5 * m_size) / PE_NUM_OP + 1;
                    calctime = vector_ops * MAC_LATENCY + SQRT_LATENCY + DIV_LATENCY;
                } 
                else if (fn == SOFTMAX_TR) { 
                    // Softmax (Max, Sub, Exp, Sum, Div)
                    int vector_ops = (3 * m_size) / PE_NUM_OP + 1;
                    calctime = vector_ops * MAC_LATENCY + EXP_LATENCY + DIV_LATENCY;
                } 
                else if (fn == ADD) { 
                    // Add (Residual Connection)
                    calctime = (m_size / PE_NUM_OP + 1) * ADD_LATENCY;
                } 
                else if (fn == EMBEDDING) { 
                    // Embedding: direct lookup, no arithmetic cost
                    calctime = 1; 
                }
                else if (fn == RMSNORM) { 
                    // RMSNorm (Square, Sum, Multiply)
                    int vector_ops = (3 * m_size) / PE_NUM_OP + 1;
                    calctime = vector_ops * MAC_LATENCY + SQRT_LATENCY + DIV_LATENCY;
                }
                else if (fn == SWIGLU) { 
                    // SwiGLU (Exp, Div, 2 MAC)
                    int vector_ops = (4 * m_size) / PE_NUM_OP + 1;
                    calctime = vector_ops * MAC_LATENCY + EXP_LATENCY + DIV_LATENCY;
                }
                else if (fn == ROPE) { 
                    // RoPE: the MAC is computing a single element (idx), not the entire m_size vector.
                    // We charge only the time to compute this specific element.
                    // 2 MACs (rotation) + CORDIC
                    int element_ops = 2; 
                    calctime = (element_ops * MAC_LATENCY) + CORDIC_LATENCY;
                }
                else if (fn == ATTENTION) { 
                    // Latency already computed in the block above
                }

                // Multiply by the PE frequency ratio
                calctime = calctime * PE_FREQ_RATIO;

                selfstatus = 4; // ready for output
                pecycle = cycles + calctime; // sync cycles

                // Send the computed result onto the NoC
                inject(2, dest_mem_id, 1, outfeature, net->vcNetwork->NI_list[NI_id], packet_id + tmp_request, id);

#ifdef Countlatency
                int stats1 = (packet_id + tmp_request)*3 + 2;
                if(stats1 < CountNum) {
                    DNN_latency[stats1][3] = pecycle;
                }
#endif
                return;
            }

            int calctime = (ch_size * m_size / PE_NUM_OP + 1) * PE_FREQ_RATIO;  //25, 10
            //int calctime = 25;

            if (current_chunk < total_chunks - 1) {
                selfstatus = 41;
                pecycle = cycles + calctime;
                return;
            }

            // activation
            if ((fn % 4) == 0) //linear
            {
                selfstatus = 4; // ready for this computation
                pecycle = cycles + calctime - PE_FREQ_RATIO; // sync cycles
            }
            else if ((fn % 4) == 1)
            {
                // activation (relu)
                // cout << "from mac " << id << " output " << outfeature << endl;
                relu(outfeature);
                selfstatus = 4; // ready for output
                pecycle = cycles + calctime; // sync cycles
            }
            else if ((fn % 4) == 2)
            {
                // activation (tanh)
                tanh(outfeature);
                selfstatus = 4; // ready for output
                pecycle = cycles + calctime; // sync cycles
            }
            else if ((fn % 4) == 3)
            {
                // activation (sigmoid)
                sigmoid(outfeature);
                selfstatus = 4; // ready for output
                pecycle = cycles + calctime; // sync cycles
            }
            else
            {
                outfeature = 0.0;
                selfstatus = 0; // back to initial state
                pecycle = cycles + 2; // sync cycles
                assert((0 < 1) && "Wrong function (fn)");
                return;
            }

            // inject
#ifndef newpooling
            inject(2, dest_mem_id, 1, outfeature, net->vcNetwork->NI_list[NI_id], packet_id + tmp_request, id);
#ifdef Countlatency
            //statistics
            stats1 = (packet_id + tmp_request)*3 + 2;
            if(stats1 < CountNum) {
                DNN_latency[stats1][3] = pecycle;
            }
#endif
            //packet_id++;
#endif
#ifdef newpooling
            if (this->npoolflag == 1){ //with pooling send out
                //only if pooling is needed
                if(this->n_tmpch >= 0){
                    for (int c : this->n_tmpm) // c is the dest id of it
                    {
                        this->tmpm = c;
                        this->tmpch = this->n_tmpch;
                        inject(2, dest_mem_id, 1, outfeature, net->vcNetwork->NI_list[NI_id], packet_id + tmp_request, id);
#ifdef Countlatency
                        if((packet_id + tmp_request)*3+2 < CountNum) {
                        DNN_latency[(packet_id + tmp_request)*3+2][6] = pecycle;}
#endif
                        //packet_id++;
                    }
                }
            }
            else{ // normal send out
                inject(2, dest_mem_id, 1, outfeature, net->vcNetwork->NI_list[NI_id], packet_id + tmp_request, id);
#ifdef Countlatency
                //statistics
                stats1 = (packet_id + tmp_request)*3 + 2;
                if(stats1 < CountNum) {
                    DNN_latency[stats1][0] = net->c_layer;
                    DNN_latency[stats1][1] = 2;
                    DNN_latency[stats1][2] = id;
                    DNN_latency[stats1][3] = pecycle;
                }
#endif
                //packet_id++;
            }
#endif

            //added to reduce transmission delay
            //this->send = 2;
            return;
        }
        else if(selfstatus == 4){
#ifndef only3type
            if(this->send == 2) // get confirmation
            {
                this->send = 0;
                if(this->routing_table.size()==0)
                {
                    this->selfstatus = 5;
                }
                else
                {
                    this->selfstatus = 0;               // back to initial state
                }
                //cout << "from mac " << this->id << " output " << this->outfeature << " " << selfstatus << endl;
                this->weight.clear();
                this->infeature.clear();
                this->inbuffer.clear();
                this->outfeature = 0.0;
                this->outfeature_vec.clear();
#ifdef newpooling
            this->npoolflag = 0;
            this->n_tmpch = -1;
            this->n_tmpm.clear();
#endif
                this->pecycle = cycles + 1; //cycles + 1
                return;
            }
            this->send = 1; // change from 1
            pecycle = cycles;
#endif
#ifdef only3type
            this->send = 0;
            if(this->routing_table.size()==0)
            {
                this->selfstatus = 5;
            }
            else
            {
                this->selfstatus = 0;               // back to initial state
            }
            this->current_chunk = 0;
            this->total_chunks = 1;
            this->psum_accumulator = 0.0f;

            //cout << "from mac " << this->id << " output " << this->outfeature << " " << selfstatus << endl;
            this->weight.clear();
            this->infeature.clear();
            this->inbuffer.clear();
            this->outfeature = 0.0;
#ifdef newpooling
            this->npoolflag = 0;
            this->n_tmpch = -1;
            this->n_tmpm.clear();
#endif
            this->pecycle = cycles + 1; //cycles + 1
            return;
#endif
        }
        else if (selfstatus == 41) {
            current_chunk++;
            selfstatus = 1;
            pecycle = cycles;
            weight.clear();
            infeature.clear();
            inbuffer.clear();
            return;
        }
    }
}

void MAC::sigmoid(float& x) // 3
{
    x = 1.0 / (1.0 + std::exp(-x));
}

void MAC::tanh(float& x)  // 2
{
    x = 2.0 / (1.0 + std::exp(-2 * x)) - 1;
}

void MAC::relu(float& x)  // 1
{
    if (x < 0) x = 0.0;
}


// Destructor
MAC::~MAC (){


}