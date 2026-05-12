/*
 * MACnet.cpp
 *
 */

#include "MACnet.hpp"
#include <algorithm>

template<class C, typename T>
bool contains(C&& c, T e) { return find(begin(c), end(c), e) != end(c); };

MACnet::MACnet (int mac_num, int t_pe_x, int t_pe_y, Model *m, VCNetwork* t_Network)
{
    macNum = mac_num;
    MAC_list.reserve(mac_num);
    pe_x = t_pe_x;
    pe_y = t_pe_y;
    cnnmodel = m;
    vcNetwork = t_Network;

    c_layer = 0;
    n_layer = cnnmodel->all_layer_size.size();
    used_pe = 0;
    o_fn = 0;
    int temp_ni_id;
    cout << "Layer in total " << n_layer << endl;
    
    causal_mask_flag = 0;
    cnoc_phase = 0; 

    for(int i=0; i<macNum; i++){ 
        temp_ni_id = i%TOT_NUM;
        MAC* nMAC = new MAC(i, this, temp_ni_id); 
        MAC_list.push_back(nMAC);
    }

    deque<int> layer_info;
    if(cnnmodel->all_layer_type[c_layer]!='i') 
    {
        cout << "err: first layer is not input" << endl;
    }
    layer_info = cnnmodel->all_layer_size[c_layer];
    in_x = layer_info[1]; 
    in_y = layer_info[2]; 
    in_ch = layer_info[3]; 
    c_layer++;

    no_x = 0; no_y = 0; nw_x = 0; nw_y = 0; no_ch = 0; npad = 0; nstride = 1; 

    layer_info = cnnmodel->all_layer_size[c_layer];
    if(cnnmodel->all_layer_type[c_layer]=='c')
    {
        w_x = layer_info[1]; w_y = layer_info[2]; o_ch = layer_info[3]; 
        w_ch = o_ch * in_ch; o_fn = layer_info[5]; pad = layer_info[6]; stride = layer_info[7]; 
        assert((in_ch == layer_info[4]) && "Input channel not correct!");
        o_x = (in_x + 2*pad - w_x) / stride + 1;
        o_y = (in_y + 2*pad - w_y) / stride + 1;
    }
    else if(cnnmodel->all_layer_type[c_layer]=='e') 
    {
        w_x = layer_info[1]; w_y = 1; o_ch = 1; w_ch = layer_info[0]; 
        o_fn = 19; pad = 0; stride = 1; o_x = w_x; o_y = in_x; 
    }

    st_w = 0;
    readyflag = 0; 
    cout << "!!MACnet created!!" << endl;
    cout << "layer" << c_layer << " created " << cnnmodel->all_layer_type[c_layer] << ' ' << in_ch << ' ' << (o_ch * o_x * o_y) << endl;

    Layer_latency.clear();
}

bool serpentine_sort(int id_a, int id_b) {
    int y_a = id_a / X_NUM;
    int x_a = id_a % X_NUM;
    
    int y_b = id_b / X_NUM;
    int x_b = id_b % X_NUM;

    // If they are on different rows, sort from top to bottom
    if (y_a != y_b) {
        return y_a < y_b; 
    } 
    // If they are on the same row, it depends on whether the row is even or odd
    else {
        if (y_a % 2 == 0) {
            return x_a < x_b; // Even rows: left to right (0 -> 7)
        } else {
            return x_a > x_b; // Odd rows: right to left (7 -> 0)
        }
    }
}

void MACnet::create_input(){
    input_table.resize(in_ch);
    int outmatsize = o_x * o_y;
    int wmatsize = w_x * w_y;
    int padded_x = in_x + 2*pad;
    int padded_y = in_y + 2*pad;
    weight_table.resize(w_ch);

    if (this->c_layer == 1) 
    {
        for(int i=0;i<in_ch;i++){
            if(pad==0) {
                input_table[i].assign(this->cnnmodel->all_data_in[i].begin(),this->cnnmodel->all_data_in[i].end());
            } else {
                input_table[i].assign(padded_x * padded_y, 0.0);
                for(int p=0; p<in_y;++p) {
                    for(int q=0;q<in_x;++q) {
                        input_table[i][(p+pad)*padded_x + (q+pad)] = this->cnnmodel->all_data_in[i][p*in_x+q];
                    }
                }
            }
        }       
    }
    else if (this->c_layer >=2)
    {
        auto& prev_out = this->layer_outputs_history[this->c_layer - 1];

        for(int i=0;i<in_ch;i++){
            if(pad==0) {
                input_table[i].assign(prev_out[i].begin(), prev_out[i].end());
            } else {
                input_table[i].assign(padded_x * padded_y, 0.0);
                for(int p=0; p<in_y;++p) {
                    for(int q=0;q<in_x;++q) {
                        input_table[i][(p+pad)*padded_x + (q+pad)] = prev_out[i][p*in_x+q];
                    }
                }
            }
        }
    }

    if (this->cnnmodel->all_layer_type[c_layer]=='c')
    {
        for(int i=0;i<o_ch;i++){
            for(int j=0;j<in_ch;j++) {
                weight_table[i*in_ch + j].assign(this->cnnmodel->all_weight_in[st_w + i].begin() + j*wmatsize,this->cnnmodel->all_weight_in[st_w + i].begin() + j*wmatsize + wmatsize); 
                weight_table[i*in_ch + j].push_back(this->cnnmodel->all_weight_in[st_w + i].back()); 
            }
        }
        st_w += o_ch;
    }
    else if (this->cnnmodel->all_layer_type[c_layer]=='f')
    {
        for(int i=0;i<w_ch;i++){
            weight_table[i].assign(this->cnnmodel->all_weight_in[st_w + i].begin(),this->cnnmodel->all_weight_in[st_w + i].end());
        }
        st_w += w_ch;
    }
    else if (this->cnnmodel->all_layer_type[c_layer] == 'm' || this->cnnmodel->all_layer_type[c_layer] == 'e' || this->cnnmodel->all_layer_type[c_layer] == 'l' || this->cnnmodel->all_layer_type[c_layer] == 'r')
    {
        for(int i=0;i<w_ch;i++){
            weight_table[i].assign(this->cnnmodel->all_weight_in[st_w + i].begin(),this->cnnmodel->all_weight_in[st_w + i].end());
        }
        st_w += w_ch;
    }
    else if (this->cnnmodel->all_layer_type[c_layer]=='p' || this->cnnmodel->all_layer_type[c_layer]=='s' || this->cnnmodel->all_layer_type[c_layer]=='a' || this->cnnmodel->all_layer_type[c_layer]=='w' || this->cnnmodel->all_layer_type[c_layer]=='o' || this->cnnmodel->all_layer_type[c_layer]=='g')
    {
        weight_table.clear();
    }

#ifdef newpooling
    if(this->cnnmodel->all_layer_type[c_layer]=='c' && this->cnnmodel->all_layer_type[c_layer+1]=='p')
    {
        nw_x = cnnmodel->all_layer_size[c_layer+1][1];
        nw_y = cnnmodel->all_layer_size[c_layer+1][2];
        no_ch = cnnmodel->all_layer_size[c_layer+1][3]; 
        npad = cnnmodel->all_layer_size[c_layer+1][4]; 
        nstride = cnnmodel->all_layer_size[c_layer+1][5]; 
        assert((o_ch == no_ch) && "Input channel not correct for merged pooling!");
        no_x = (o_x + 2*npad - nw_x) / nstride + 1;
        no_y = (o_y + 2*npad - nw_y) / nstride + 1;
        outmatsize = no_x * no_y;
    }
#endif
    output_table.resize(o_ch);
    for(int i=0;i<o_ch;i++){
        output_table[i].assign(outmatsize, 0.0);
    }
}

void MACnet::mapping(int neuronnum){
    this->mapping_table.clear();
    this->mapping_table.resize(macNum);
    int head_dim = 1;
    if (this->cnnmodel->all_layer_type[c_layer] == 't') {
        int q_dim = this->cnnmodel->all_layer_size[c_layer][1];
        int n_heads = this->cnnmodel->all_layer_size[c_layer][3];
        head_dim = q_dim / n_heads;
    }
    int j = 0;
    int mac_iter = 0;
    while (j < neuronnum){
        if (this->cnnmodel->all_layer_type[c_layer] == 's' && this->causal_mask_flag == 1) {
            if ((j % o_x) > (j / o_x)) { j++; continue; }
        }
        while (contains(dest_list, mac_iter % TOT_NUM)) {
            mac_iter = (mac_iter + 1) % macNum;
        }
        this->mapping_table[mac_iter].push_back(j);
        j++; 
        if (this->cnnmodel->all_layer_type[c_layer] == 't') {
            if (j % head_dim == 0) mac_iter = (mac_iter + 1) % macNum;
        } else {
            mac_iter = (mac_iter + 1) % macNum;
        }
    }
}

void MACnet::ymapping(int neuronnum){
    this->mapping_table.clear();
    this->mapping_table.resize(macNum);
    int head_dim = 1;
    if (this->cnnmodel->all_layer_type[c_layer] == 't') {
        int q_dim = this->cnnmodel->all_layer_size[c_layer][1];
        int n_heads = this->cnnmodel->all_layer_size[c_layer][3];
        head_dim = q_dim / n_heads;
    }
    int npos[macNum];
    for (int c = 0; c < PE_X_NUM; c++) {
        for (int r = 0; r < PE_Y_NUM; r++) {
            npos[c * PE_Y_NUM + r] = r * PE_X_NUM + c;
        }
    }
    int j = 0;
    int mac_iter = 0;
    while (j < neuronnum){
        if (this->cnnmodel->all_layer_type[c_layer] == 's' && this->causal_mask_flag == 1) {
            if ((j % o_x) > (j / o_x)) { j++; continue; }
        }
        int k = npos[mac_iter];
        int temp_i = k % TOT_NUM;
        if (contains(dest_list, temp_i)) {
            mac_iter = (mac_iter + 1) % macNum;
            continue;
        }
        this->mapping_table[k].push_back(j);
        j++; 
        if (this->cnnmodel->all_layer_type[c_layer] == 't') {
            if (j % head_dim == 0) mac_iter = (mac_iter + 1) % macNum;
        } else {
            mac_iter = (mac_iter + 1) % macNum;
        }
    }
}

void MACnet::rmapping(int neuronnum){
    this->mapping_table.clear();
    this->mapping_table.resize(macNum);
    int head_dim = 1;
    if (this->cnnmodel->all_layer_type[c_layer] == 't') {
        int q_dim = this->cnnmodel->all_layer_size[c_layer][1];
        int n_heads = this->cnnmodel->all_layer_size[c_layer][3];
        head_dim = q_dim / n_heads;
    }
    unsigned seed = 0; 
    vector<int> npos;
    for (int c=0; c<macNum; c++) {npos.push_back(c);}
    shuffle(npos.begin(), npos.end(), default_random_engine(seed));
    int j = 0;
    int mac_iter = 0;
    while (j < neuronnum){
        if (this->cnnmodel->all_layer_type[c_layer] == 's' && this->causal_mask_flag == 1) {
            if ((j % o_x) > (j / o_x)) { j++; continue; }
        }
        int k = npos[mac_iter];
        int temp_i = k % TOT_NUM;
        if (contains(dest_list, temp_i)) {
            mac_iter = (mac_iter + 1) % macNum;
            continue;
        }
        this->mapping_table[k].push_back(j);
        j++; 
        if (this->cnnmodel->all_layer_type[c_layer] == 't') {
            if (j % head_dim == 0) mac_iter = (mac_iter + 1) % macNum;
        } else {
            mac_iter = (mac_iter + 1) % macNum;
        }
    }
}

// void MACnet::cNoC_mapping(int task_num) {
//     cnoc_compute_path.clear();
//     this->mapping_table.clear();
//     this->mapping_table.resize(macNum);
    
//     std::vector<int> avail_routers;
//     // Vector to track how much SRAM (in number of floats) is occupied in each router
//     std::vector<int> router_sram_usage(TOT_NUM, 0); 

//     for (int i = 0; i < TOT_NUM; i++) {
//         // this->vcNetwork->router_list[i]->local_weights.clear();
//         this->vcNetwork->router_list[i]->clearSRAM();

//         this->vcNetwork->router_list[i]->assigned_tasks.clear(); 
//         if (!contains(dest_list, i)) {
//             avail_routers.push_back(i);
//         }
//     }
    
//     for (int t = 0; t < task_num; t++) {
//         // Determine the weight memory footprint of the current task
//         int w_idx = t % (this->weight_table.empty() ? 1 : this->weight_table.size());
//         int required_sram = this->weight_table.empty() ? 1 : this->weight_table[w_idx].size();

//         bool task_assigned = false;

//         // Search for an available router with enough SRAM space
//         for (size_t i = 0; i < avail_routers.size(); i++) {
//             // Round-Robin logic: try the router that "should" take this task first
//             int r_idx = (t + i) % avail_routers.size();
//             int r_id = avail_routers[r_idx];

//             if (router_sram_usage[r_id] + required_sram <= ROUTER_SRAM_LIMIT) {
//                 // Sufficient space: assign the task to the router
//                 this->vcNetwork->router_list[r_id]->assigned_tasks.push_back(t);
//                 router_sram_usage[r_id] += required_sram;
                
//                 // If it's not already in the compute path, add it
//                 if (std::find(cnoc_compute_path.begin(), cnoc_compute_path.end(), r_id) == cnoc_compute_path.end()) {
//                     cnoc_compute_path.push_back(r_id);
//                 }
                
//                 task_assigned = true;
//                 break; // Exit the router search loop, move to the next task
//             }
//         }

//         // If no router has enough space, trigger a Resource Overflow
//         if (!task_assigned) {
//             cout << "\n[FATAL ERROR] ----------------------------------------------------" << endl;
//             cout << "ROUTER_SRAM_LIMIT (" << ROUTER_SRAM_LIMIT << " floats) exceeded!" << endl;
//             cout << "The NoC does not have enough Routers to spatially distribute" << endl;
//             cout << "the weights of layer " << c_layer << " (" << cnnmodel->all_layer_type[c_layer] << ")." << endl;
//             cout << "Solutions: " << endl;
//             cout << " 1. Increase ROUTER_SRAM_LIMIT in parameters.hpp" << endl;
//             cout << " 2. Increase the NoC dimensions (e.g., MemNode18, MemNode32)" << endl;
//             cout << "------------------------------------------------------------------\n" << endl;
//             exit(1); 
//         }
//     }

//     // Topological sort based on the physical layout of the NoC to minimize routing conflicts and potential deadlocks
//     std::sort(cnoc_compute_path.begin(), cnoc_compute_path.end(), serpentine_sort);
    
//     cnoc_phase = 1; 
// }

void MACnet::cNoC_mapping(int task_num) {
    cnoc_compute_path.clear();
    this->mapping_table.clear();
    this->mapping_table.resize(macNum);
    
    std::vector<int> avail_routers;
    std::vector<int> router_sram_usage(TOT_NUM, 0); 

    for (int i = 0; i < TOT_NUM; i++) {
        this->vcNetwork->router_list[i]->clearSRAM();
        this->vcNetwork->router_list[i]->assigned_tasks.clear(); 
        if (!contains(dest_list, i)) {
            avail_routers.push_back(i);
        }
    }
    
    this->chunk_start_task_idx = this->last_mapped_task_idx;
    
    bool sram_full_globally = false;

    for (int t = this->last_mapped_task_idx; t < task_num; t++) {
        int w_idx = t % (this->weight_table.empty() ? 1 : this->weight_table.size());
        int required_sram = this->weight_table.empty() ? 1 : this->weight_table[w_idx].size();

        bool task_assigned = false;

        for (size_t i = 0; i < avail_routers.size(); i++) {
            int r_idx = (t + i) % avail_routers.size();
            int r_id = avail_routers[r_idx];

            if (router_sram_usage[r_id] + required_sram <= ROUTER_SRAM_LIMIT) {
                this->vcNetwork->router_list[r_id]->assigned_tasks.push_back(t);
                router_sram_usage[r_id] += required_sram;
                
                if (std::find(cnoc_compute_path.begin(), cnoc_compute_path.end(), r_id) == cnoc_compute_path.end()) {
                    cnoc_compute_path.push_back(r_id);
                }
                
                task_assigned = true;
                break;
            }
        }

        if (!task_assigned) {
            sram_full_globally = true;
            this->last_mapped_task_idx = t;
            break;
        }
    }

    if (!sram_full_globally) {
        this->last_mapped_task_idx = task_num;
    }

    this->tasks_in_current_chunk = this->last_mapped_task_idx - this->chunk_start_task_idx;

    std::sort(cnoc_compute_path.begin(), cnoc_compute_path.end(), serpentine_sort);
    
    this->current_chunk_idx++;
    this->cnoc_phase = 1; 
}

void MACnet::inject_cNoC_traffic() {
    int mem_id = dest_list[0]; 
    NI* mem_ni = this->vcNetwork->NI_list[mem_id];

    if (cnoc_phase == 1) {
        int w_idx_path = 0;

        for (int router_id : cnoc_compute_path) {
            Message dist_msg = Message(); 
            dist_msg.source_id = mem_id;
            dist_msg.NI_id = mem_id;
            dist_msg.mac_id = mem_id;
            dist_msg.destination = router_id; 
            dist_msg.type = 4; 
            dist_msg.compute_op = o_fn;
            dist_msg.out_cycle = cycles; 
            dist_msg.signal_id = packet_id;

            auto router = this->vcNetwork->router_list[router_id];
            if (!this->weight_table.empty()) {
                for (int task_idx : router->assigned_tasks) {
                    int w_idx = task_idx % this->weight_table.size();
                    dist_msg.data.insert(dist_msg.data.end(), this->weight_table[w_idx].begin(), this->weight_table[w_idx].end());
                }
            } else {
                dist_msg.data.push_back(1.0f); 
            }
            
            dist_msg.data_length = dist_msg.data.size();
            
            Packet* p = Packet::allocate(std::move(dist_msg), X_NUM, mem_ni->NI_num);
            // p->send_out_time = cycles;
            // p->in_net_time = cycles;
            // mem_ni->packetBuffer_list[p->vnet]->enqueue(p);

            int memory_latency = std::ceil(dist_msg.data_length * MEM_read_delay) + CACHE_DELAY;
            p->send_out_time = cycles + memory_latency; 
            
            p->in_net_time = cycles;
            mem_ni->packetBuffer_list[p->vnet]->enqueue(p);

            w_idx_path++;
        }
        cnoc_phase = 2; 
    }
    else if (cnoc_phase == 2) {
        // Send a packet cNoC for EACH token in the sequence (in_y)
        for (int y = 0; y < this->in_y; y++) {
            Message comp_msg = Message();
            comp_msg.source_id = mem_id;
            comp_msg.NI_id = mem_id;
            comp_msg.mac_id = mem_id;
            comp_msg.destination = mem_id; 
            comp_msg.type = 5; 
            comp_msg.compute_op = o_fn;
            comp_msg.out_cycle = cycles; 
            comp_msg.signal_id = packet_id + y;
            
            // Use the sequence_id to let the memory know which row to save the result in.
            comp_msg.sequence_id = y; 
            
            comp_msg.chunk_start_idx = this->chunk_start_task_idx;
            
            comp_msg.routing_path.assign(cnoc_compute_path.begin(), cnoc_compute_path.end());
            comp_msg.routing_path.push_back(mem_id);
            
            // start and end indices for token 'y'
            int token_start = y * this->in_x;
            int token_end   = (y + 1) * this->in_x;
            
            // Select insertion mode based on operation
            comp_msg.data.reserve(this->in_x * 2 + this->tasks_in_current_chunk); 
            if (o_fn == 18 || o_fn == 21|| o_fn == 24) { // 18 = ADD, 21 = SWIGLU, 24 = GeGLU
                bool has_residual = false;
                std::vector<float> secondary_data;

                if (o_fn == 18) {
                    int residual_source_id = this->cnnmodel->all_layer_size[c_layer][1]; 
                    if (layer_outputs_history.find(residual_source_id) != layer_outputs_history.end() && !layer_outputs_history[residual_source_id].empty()) {
                        secondary_data = layer_outputs_history[residual_source_id][0];
                        has_residual = true;
                    }
                }

                for (int k = 0; k < this->tasks_in_current_chunk; k++) {
                    int base_idx = y * this->in_x + this->chunk_start_task_idx + k;
                    
                    // First operand (x for ADD, gate for SWIGLU)
                    comp_msg.data.push_back(this->input_table[0][base_idx]); 
                    
                    // Second operand
                    if (o_fn == 18 && has_residual) {
                        comp_msg.data.push_back(secondary_data[base_idx]); 
                    } else if (o_fn == 21 || o_fn == 24) {
                        // For SWIGLU and GeGLU, the "up" projection is offset by o_x
                        comp_msg.data.push_back(this->input_table[0][base_idx + o_x]); 
                    } else {
                        comp_msg.data.push_back(0.0f); // Safety fallback
                    }
                }
            } else {
                // Normal operations with 1 operand per token (e.g., MATMUL)
                if (this->input_table.size() > 0 && this->input_table[0].size() >= token_end) {
                    comp_msg.data.reserve(token_end - token_start + this->tasks_in_current_chunk);
                    comp_msg.data.assign(
                        this->input_table[0].begin() + token_start, 
                        this->input_table[0].begin() + token_end
                    );
                }
            }
            
            if (o_fn == MATMUL || o_fn == ATTENTION) {
                comp_msg.psum_offset = comp_msg.data.size();
                if (o_fn == ATTENTION) {
                    comp_msg.k_dim = cnnmodel->all_layer_size[c_layer][2];
                }          
#if USE_BIAS
                if (o_fn == MATMUL) {
                    for(size_t b = 0; b < this->tasks_in_current_chunk; b++) {
                        comp_msg.data.push_back(this->weight_table[this->chunk_start_task_idx + b].back()); 
                    }
                } else {
                    comp_msg.data.insert(comp_msg.data.end(), this->tasks_in_current_chunk, 0.0f); 
                }
#else
                comp_msg.data.insert(comp_msg.data.end(), this->tasks_in_current_chunk, 0.0f); 
#endif
                
            } else {
                // For ADD, SWIGLU, etc., we perform pure IN-PLACE execution
                comp_msg.psum_offset = 0;
            }
            
            comp_msg.data_length = comp_msg.data.size();
            
            Packet* p = Packet::allocate(std::move(comp_msg), X_NUM, mem_ni->NI_num);
            // p->send_out_time = cycles;
            // p->in_net_time = cycles;
            // mem_ni->packetBuffer_list[p->vnet]->enqueue(p);

            int memory_latency = std::ceil(comp_msg.data_length * MEM_read_delay) + CACHE_DELAY;
            p->send_out_time = cycles + memory_latency; 
            
            p->in_net_time = cycles;
            mem_ni->packetBuffer_list[p->vnet]->enqueue(p);
        }
        
        cnoc_phase = 3 + this->in_y;
    }
}

void MACnet::checkStatus()
{
    if(readyflag == 0) 
    {
        this->create_input();
        
        if (this->cnnmodel->all_layer_type[c_layer] == 'l') {
            precalc_mean.assign(o_y, 0.0);
            precalc_var.assign(o_y, 0.0);
            for(int tmpy = 0; tmpy < o_y; tmpy++) {
                float mean = 0.0, var = 0.0;
                for(int j = 0; j < in_x; j++) mean += this->input_table[0][tmpy*in_x + j];
                mean /= in_x;
                for(int j = 0; j < in_x; j++) var += (this->input_table[0][tmpy*in_x + j] - mean)*(this->input_table[0][tmpy*in_x + j] - mean);
                var /= in_x;
                precalc_mean[tmpy] = mean;
                precalc_var[tmpy] = var;
            }
        } else if (this->cnnmodel->all_layer_type[c_layer] == 'r') {
            precalc_rms.assign(o_y, 0.0);
            for(int tmpy = 0; tmpy < o_y; tmpy++) {
                float sum_sq = 0.0;
                for(int j = 0; j < in_x; j++) sum_sq += this->input_table[0][tmpy*in_x + j] * this->input_table[0][tmpy*in_x + j];
                precalc_rms[tmpy] = std::sqrt((sum_sq / in_x) + 1e-5);
            }
        }

        int task_num = (o_ch * o_x * o_y);

        this->total_tasks_in_layer = (o_ch * o_x);
        this->last_mapped_task_idx = 0;
        this->current_chunk_idx = 0;
        this->tiling_active = true;

#ifdef cNoC_MODE
        char l_type = this->cnnmodel->all_layer_type[c_layer];
        if (l_type == 'm' || l_type == 'a' || l_type == 'w'|| l_type == 'g' || l_type == 't')
        {
            this->cNoC_mapping(total_tasks_in_layer);
        } else {
            this->mapping(task_num); 
            this->tiling_active = false;
        }
#else
        this->mapping(task_num);
        this->tiling_active = false;
#endif
        for(int i=0; i<macNum; i++)
        {
            if(mapping_table[i].size() == 0)
            {
                this->MAC_list[i]->selfstatus = 5;
#ifdef only3type
                this->MAC_list[i]->send = 3;
#endif
            }
            else
            {
                this->MAC_list[i]->routing_table.assign(mapping_table[i].begin(),mapping_table[i].end());
            }
            // Reset della cache hardware del singolo PE
            this->MAC_list[i]->local_sram_usage = 0;
            this->MAC_list[i]->kv_cache.clear(); 
            this->MAC_list[i]->cached_score_row = -1;
            this->MAC_list[i]->cached_score_head = -1;
        }

        readyflag = 1; 
        return;
    }
    
    for(int i=0; i<macNum; i++){
        if(MAC_list[i]->selfstatus != 5) {
            readyflag = 1;
            return;
        }
#ifdef only3type
        else {
            if(MAC_list[i]->send != 3) {
                readyflag = 1;
                return;
            }
        }
#endif
    }
    
#ifdef cNoC_MODE
    if (cnoc_phase != 0) {
        readyflag = 1;
        return;
    }
#endif
    
    deque<int> layer_info;
    in_x = o_x; in_y = o_y; in_ch = o_ch; 
    
    layer_outputs_history[c_layer] = std::move(output_table);

    std::vector<int> layers_to_delete;
    for (auto const& item : layer_outputs_history) {
        int saved_id = item.first;
        bool is_needed = false;

        for (int future_l = c_layer + 1; future_l < n_layer; future_l++) {
            if (this->cnnmodel->all_layer_type[future_l] == 'a') { 
                int required_src = this->cnnmodel->all_layer_size[future_l][1];
                if (required_src == saved_id) {
                    is_needed = true;
                    break;
                }
            }
        }
        if (!is_needed && saved_id != c_layer) {
            layers_to_delete.push_back(saved_id);
        }
    }
    for (int id_del : layers_to_delete) {
        layer_outputs_history.erase(id_del);
    }

    #ifdef newpooling
        if(this->cnnmodel->all_layer_type[c_layer]=='c' && this->cnnmodel->all_layer_type[c_layer+1]=='p')
        {
            in_ch = no_ch; in_x = no_x; in_y = no_y; c_layer++;
        }
    #endif

    if (vcNetwork != NULL) {
        vcNetwork->clearAllRouterSRAM();
    }

    c_layer++; 
    if(c_layer == n_layer)
    {
        cout << "All finished! at cycle " << cycles << endl;
        output_table = layer_outputs_history[c_layer - 1];
        Layer_latency.push_back(cycles);
        readyflag = 2;
        packet_id = packet_id + o_ch*o_x*o_y;
        return;
    }
    else
    {
        cout << "Layer finished " << (c_layer-1) << " at cycle " << cycles << endl;
        Layer_latency.push_back(cycles);
        packet_id = packet_id + o_ch*o_x*o_y;
    }

    for(int ir=0; ir<TOT_NUM; ir++){
        this->vcNetwork->router_list[ir]->rr_port = 0;
        this->vcNetwork->NI_list[ir]->rr_buffer = 0;
        this->vcNetwork->NI_list[ir]->rr_priority_record = 0;
        for (int ip=0; ip<5;ip++)
        {
            this->vcNetwork->router_list[ir]->in_port_list[ip]->rr_record = 0;
            this->vcNetwork->router_list[ir]->in_port_list[ip]->rr_priority_record = 0;
        }
    }

    layer_info = cnnmodel->all_layer_size[c_layer];
    if(cnnmodel->all_layer_type[c_layer]=='c')  
    {
        w_x = layer_info[1]; w_y = layer_info[2]; o_ch = layer_info[3]; 
        w_ch = o_ch * in_ch; o_fn = layer_info[5]; pad = layer_info[6]; stride = layer_info[7]; 
        assert((in_ch == layer_info[4]) && "Input channel not correct!");
        o_x = (in_x + 2*pad - w_x) / stride + 1;
        o_y = (in_y + 2*pad - w_y) / stride + 1;
    }
    else if(cnnmodel->all_layer_type[c_layer]=='f')  
    {
        in_x = layer_info[0]; in_ch = 1; in_y = 1; w_x = layer_info[0]; 
        w_y = 1; o_ch = 1; w_ch = layer_info[1]; o_fn = layer_info[2] + 4; pad = 0; stride = 1;
        assert((in_x == w_x) && "Input channel not correct!");
        o_x = layer_info[1]; o_y = 1;
        if(this->output_table.size() > 1) 
        {
            vector<float> temp_out_table;
            for(int z = 0; z < this->output_table.size(); z++)
            {
                temp_out_table.insert(temp_out_table.end(), this->output_table[z].begin(), this->output_table[z].end());
            }
            this->output_table.resize(1);
            this->output_table[0].assign(temp_out_table.begin(), temp_out_table.end());
        }
    }
    else if(cnnmodel->all_layer_type[c_layer]=='p') 
    {
        w_x = layer_info[1]; w_y = layer_info[2]; o_ch = layer_info[3]; 
        pad = layer_info[4]; stride = layer_info[5]; w_ch = 0; o_fn = 8; 
        if (layer_info[6] == 2) {o_fn = 12;} 
        assert((in_ch == o_ch) && "Input channel not correct!");
        o_x = (in_x + 2*pad - w_x) / stride + 1; o_y = (in_y + 2*pad - w_y) / stride + 1;
    }
    else if(cnnmodel->all_layer_type[c_layer]=='e') { in_x = layer_info[0]; in_ch = 1; w_x = layer_info[1]; w_y = 1; o_ch = 1; w_ch = layer_info[0]; o_fn = EMBEDDING; o_x = w_x; }
    else if(cnnmodel->all_layer_type[c_layer]=='m') { in_x = layer_info[0]; in_ch = 1; w_x = layer_info[0]; w_y = 1; o_ch = 1; w_ch = layer_info[1]; o_fn = MATMUL; o_x = layer_info[1]; o_y = in_y; }
    else if(cnnmodel->all_layer_type[c_layer]=='l') { in_x = layer_info[0]; in_ch = 1; w_x = layer_info[0]; w_y = 1; o_ch = 1; w_ch = 2; o_fn = LAYERNORM; o_x = in_x; o_y = in_y; }
    else if(cnnmodel->all_layer_type[c_layer]=='r') { in_x = layer_info[0]; in_ch = 1; w_x = layer_info[0]; w_y = 1; o_ch = 1; w_ch = 1; o_fn = RMSNORM; o_x = in_x; o_y = in_y; }
    else if(cnnmodel->all_layer_type[c_layer]=='s') { in_x = layer_info[0]; causal_mask_flag = layer_info[1]; in_ch = 1; w_x = in_x; w_y = 1; o_ch = 1; w_ch = 0; o_fn = SOFTMAX_TR; o_x = in_x; o_y = in_x; }
    else if(cnnmodel->all_layer_type[c_layer]=='a') { in_x = layer_info[0]; in_ch = 1; w_x = in_x; w_y = 1; o_ch = 1; w_ch = 0; o_fn = ADD; o_x = in_x; o_y = in_y; }
    else if(cnnmodel->all_layer_type[c_layer]=='w') { in_x = layer_info[0] * 2; in_ch = 1; w_x = in_x; w_y = 1; o_ch = 1; w_ch = 0; o_fn = SWIGLU; o_x = layer_info[0]; o_y = in_y; }
    else if(cnnmodel->all_layer_type[c_layer]=='g') { in_x = layer_info[0] * 2; in_ch = 1; w_x = in_x; w_y = 1; o_ch = 1; w_ch = 0; o_fn = GEGLU; o_x = layer_info[0]; o_y = in_y; }
    else if(cnnmodel->all_layer_type[c_layer]=='o') { in_x = layer_info[0]; in_ch = 1; w_x = in_x; w_y = 1; o_ch = 1; w_ch = 0; o_fn = ROPE; o_x = in_x; o_y = in_y; }
    else if(cnnmodel->all_layer_type[c_layer]=='t') { in_x = layer_info[0]; in_ch = 1; w_x = in_x; w_y = 1; o_ch = 1; w_ch = 0; o_fn = ATTENTION; o_x = layer_info[1]; }

    readyflag = 0;
    for(int i=0; i<macNum; i++){
        MAC_list[i]->selfstatus = 0;
        MAC_list[i]->pecycle = cycles;
    }   
    return;
}

void MACnet::runOneStep()
{
#ifdef cNoC_MODE
    if (cnoc_phase == 1 || cnoc_phase == 2) {
        inject_cNoC_traffic();
    }
#endif

    MAC * tmpMAC;
    NI * tmpNI;
    Packet * tmpPacket;
    for(int i=0; i<macNum; i++){ 
        MAC_list[i]->runOneStep();
    }

    int pbuffersize;
    int src;
    int pid;
    int mem_id;
    int src_mac;
    
    for(int memidx=0;memidx<MEM_NODES;memidx++)
    {
        mem_id = dest_list[memidx];
        tmpNI = this->vcNetwork->NI_list[mem_id];
        
        for (auto it = tmpNI->packet_buffer_out[0].begin(); it != tmpNI->packet_buffer_out[0].end(); ) {
            tmpPacket = *it;
            
#ifdef cNoC_MODE
            if (tmpPacket->message.type == 5 && tmpPacket->message.out_cycle <= cycles) {
                int y = tmpPacket->message.sequence_id;
                int p_offset = tmpPacket->message.psum_offset;

                int c_offset = tmpPacket->message.chunk_start_idx;
                int k_count = this->tasks_in_current_chunk;

                if (tmpPacket->message.compute_op == ATTENTION) {                      
                    double final_denominator = tmpPacket->message.running_sum;
                    double epsilon = 1e-9;
                    if (final_denominator < epsilon) final_denominator = epsilon;

                    for(size_t k = 0; k < k_count; k++) { 
                        tmpPacket->message.data[p_offset + k] /= final_denominator;
                    }
                }
                
                // Saving data by extracting from 'data'
                for(size_t k = 0; k < k_count; k++) {
                    int out_idx = (y * o_x) + c_offset + k;                                                              
                    if (out_idx < this->output_table[0].size()) {
                        
                        if (tmpPacket->message.compute_op == ADD) {
                            // Add operation is linear, it has been computed in-transit in the router
                            this->output_table[0][out_idx] = tmpPacket->message.data[k * 2];
                        } 
                        else if (tmpPacket->message.compute_op == SWIGLU || tmpPacket->message.compute_op == GEGLU) {
                            // Non linear calculation at the terminal node (Terminal Node Concept)
                            // We read the raw data transported by the cNoC and compute here
                            float gate = tmpPacket->message.data[k * 2];
                            float up = tmpPacket->message.data[k * 2 + 1];
                            if (tmpPacket->message.compute_op == SWIGLU) {
                                float silu = gate * (1.0f / (1.0f + std::exp(-gate)));
                                this->output_table[0][out_idx] = silu * up;
                            } else {
                                // GeGLU Formula: 0.5 * gate * (1 + erf(gate / sqrt(2))) * up
                                float gelu = 0.5f * gate * (1.0f + std::erf(gate / 1.41421356f));
                                this->output_table[0][out_idx] = gelu * up;
                            }
                        }
                        else {
                            // MATMUL, LINEAR and ATTENTION
                            this->output_table[0][out_idx] = tmpPacket->message.data[p_offset + k]; 
                        }
                    }
                }
                
                cnoc_phase--;
                
                if (cnoc_phase == 3) { 
                    if (this->tiling_active && this->last_mapped_task_idx < this->total_tasks_in_layer) {
                        
                        this->vcNetwork->clearAllRouterSRAM();
                        this->cNoC_mapping(this->total_tasks_in_layer);
                    } 
                    else {
                        cnoc_phase = 0; 
                        for(int m=0; m<macNum; m++){
                            MAC_list[m]->selfstatus = 5; 
#ifdef only3type
                            MAC_list[m]->send = 3;
#endif
                        }
                    }
                }
                
                it = tmpNI->packet_buffer_out[0].erase(it);
                Packet::release(tmpPacket);
                continue;
            }
#endif

            if(tmpPacket->message.type != 0 || tmpPacket->message.out_cycle >= cycles)
            {
                ++it;
                continue;
            }
            src = tmpPacket->message.source_id;
            pid = tmpPacket->message.signal_id;
            src_mac = tmpPacket->message.mac_id;

#ifdef Countlatency
            if(pid*3 < CountNum) {
                DNN_latency[pid*3][4] = tmpPacket->send_out_time;
                DNN_latency[pid*3][7] = cycles;
            }
            if(pid*3+1 < CountNum) {
                DNN_latency[pid*3+1][1] = 1;
                DNN_latency[pid*3+1][2] = src_mac;
                DNN_latency[pid*3+1][3] = cycles;
            }
#endif
            tmpMAC = MAC_list[src_mac];
            
            if(this->cnnmodel->all_layer_type[c_layer]=='c'){ 
                if(tmpMAC->selfstatus == 2) 
                {
                    tmpMAC->tmpch = tmpMAC->request / (o_x*o_y); 
                    tmpMAC->tmpm  = tmpMAC->request % (o_x*o_y); 
                    tmpMAC->npoolflag = 0;
                    int tmpx = tmpMAC->tmpm % o_x;
                    int tmpy = tmpMAC->tmpm / o_x;
                    tmpMAC->inbuffer.clear();
                    tmpMAC->inbuffer.push_back(o_fn);
                    tmpMAC->inbuffer.push_back(in_ch);
                    tmpMAC->inbuffer.push_back(w_x * w_y);
                    
                    for (int k=0; k<in_ch; k++) {
                        for (int p=0; p<w_y; p++) {
                            tmpMAC->inbuffer.insert(tmpMAC->inbuffer.end(), this->input_table[k].begin() + (tmpy*stride+p)*(in_x+2*pad) + tmpx*stride, this->input_table[k].begin() + (tmpy*stride+p)*(in_x+2*pad) + tmpx*stride + w_x);
                        }
                    }
                    for (int k=0; k<in_ch; k++) {
                        tmpMAC->inbuffer.insert(tmpMAC->inbuffer.end(),this->weight_table[tmpMAC->tmpch*in_ch+k].begin(), this->weight_table[tmpMAC->tmpch*in_ch+k].end()-1);
                    }
                    tmpMAC->inbuffer.push_back(this->weight_table[tmpMAC->tmpch*in_ch].back()); 
#ifdef newpooling
                    if (this->cnnmodel->all_layer_type[c_layer+1]=='p')
                    {
                        tmpMAC->npoolflag = 1;
                        int n_tmpx; int n_tmpy;
                        if(tmpx >= (no_x-1)*nstride+nw_x || tmpy >= (no_y-1)*nstride+nw_y) {
                            tmpMAC->n_tmpch = -1;
                            tmpMAC->n_tmpm.clear();
                            tmpMAC->inbuffer.assign(4, 10);
                        } else {
                            tmpMAC->n_tmpch = tmpMAC->tmpch; 
                            for(n_tmpx=0; n_tmpx<no_x;n_tmpx++){
                                for(n_tmpy=0; n_tmpy<no_y;n_tmpy++){
                                    if(tmpx >= n_tmpx*nstride && tmpx < n_tmpx*nstride + nw_x && tmpy >= n_tmpy*nstride && tmpy < n_tmpy*nstride + nw_y)
                                    {tmpMAC->n_tmpm.push_back(n_tmpx+n_tmpy*no_x);}
                                }
                            }
                        }
                    }
#endif
                    MAC_list[mem_id]->pecycle = cycles + ceil((in_ch * w_x * w_y * 2 + 1) * MEM_read_delay)  + CACHE_DELAY;
                    MAC_list[mem_id]->inject(1,src,tmpMAC->inbuffer.size(),o_fn,vcNetwork->NI_list[mem_id],pid,src_mac);
                }
            }
            else if (this->cnnmodel->all_layer_type[c_layer]=='p') 
            {
                if(tmpMAC->selfstatus == 2) 
                {
                    tmpMAC->tmpch = tmpMAC->request / (o_x*o_y); 
                    tmpMAC->tmpm  = tmpMAC->request % (o_x*o_y); 
                    int tmpx = tmpMAC->tmpm % o_x;
                    int tmpy = tmpMAC->tmpm / o_x;
                    tmpMAC->inbuffer.clear();
                    tmpMAC->inbuffer.push_back(o_fn); 
                    tmpMAC->inbuffer.push_back(w_x * w_y);
                    for (int p=0; p<w_y; p++) {
                        tmpMAC->inbuffer.insert(tmpMAC->inbuffer.end(), this->input_table[tmpMAC->tmpch].begin() + (tmpy*stride+p)*(in_x+2*pad) + tmpx*stride, this->input_table[tmpMAC->tmpch].begin() + (tmpy*stride+p)*(in_x+2*pad) + tmpx*stride + w_x);
                    }
                    MAC_list[mem_id]->pecycle = cycles + ceil(w_x * w_y * MEM_read_delay) + CACHE_DELAY;
                    MAC_list[mem_id]->inject(1,src,tmpMAC->inbuffer.size(),o_fn,vcNetwork->NI_list[mem_id],pid,src_mac);
                }
            }
            else if (this->cnnmodel->all_layer_type[c_layer]=='f'){ 
                if(tmpMAC->selfstatus == 2) 
                {
                    tmpMAC->tmpch = 0; 
                    tmpMAC->tmpm  = tmpMAC->request; 
                    tmpMAC->npoolflag = 0;
                    tmpMAC->inbuffer.clear();
                    tmpMAC->inbuffer.push_back(o_fn);
                    tmpMAC->inbuffer.push_back(w_x * w_y);
                    tmpMAC->inbuffer.insert(tmpMAC->inbuffer.end(), this->input_table[0].begin(), this->input_table[0].end());
                    tmpMAC->inbuffer.insert(tmpMAC->inbuffer.end(),this->weight_table[tmpMAC->tmpm].begin(), this->weight_table[tmpMAC->tmpm].end()); 
                    
                    MAC_list[mem_id]->pecycle = cycles + ceil((w_x * w_y * 2 + 1) * MEM_read_delay) + CACHE_DELAY;
                    MAC_list[mem_id]->inject(1,src,tmpMAC->inbuffer.size(),o_fn,vcNetwork->NI_list[mem_id],pid,src_mac);
                }
            }
            else if (this->cnnmodel->all_layer_type[c_layer]=='m' || this->cnnmodel->all_layer_type[c_layer]=='l' || 
                    this->cnnmodel->all_layer_type[c_layer]=='s' || this->cnnmodel->all_layer_type[c_layer]=='a' || 
                    this->cnnmodel->all_layer_type[c_layer]=='e' || this->cnnmodel->all_layer_type[c_layer]=='r' || 
                    this->cnnmodel->all_layer_type[c_layer]=='w' || this->cnnmodel->all_layer_type[c_layer]=='o' || 
                    this->cnnmodel->all_layer_type[c_layer]=='t' || this->cnnmodel->all_layer_type[c_layer]=='g')
            {
                if(tmpMAC->selfstatus == 2) 
                {
                    tmpMAC->tmpch = 0; 
                    tmpMAC->tmpm  = tmpMAC->request; 
                    tmpMAC->npoolflag = 0;
                    tmpMAC->inbuffer.clear();

                    int tmpy = tmpMAC->tmpm / o_x;
                    int tmpx = tmpMAC->tmpm % o_x;

                    if (o_fn == MATMUL) { 
                        tmpMAC->inbuffer.push_back(o_fn);
                        tmpMAC->inbuffer.push_back(in_x);
                        tmpMAC->inbuffer.insert(tmpMAC->inbuffer.end(), this->input_table[0].begin() + tmpy*in_x, this->input_table[0].begin() + tmpy*in_x + in_x); 
                        tmpMAC->inbuffer.insert(tmpMAC->inbuffer.end(), this->weight_table[tmpx].begin(), this->weight_table[tmpx].end()); 
                    }
                    else if (o_fn == LAYERNORM) { 
                        tmpMAC->inbuffer.push_back(o_fn);
                        tmpMAC->inbuffer.push_back(in_x);
                        float mean = this->precalc_mean[tmpy];
                        float var = this->precalc_var[tmpy];
                        tmpMAC->inbuffer.push_back(mean);
                        tmpMAC->inbuffer.push_back(var);
                        tmpMAC->inbuffer.push_back(this->input_table[0][tmpy*in_x + tmpx]); 
                        tmpMAC->inbuffer.push_back(this->weight_table[0][tmpx]);            
                        tmpMAC->inbuffer.push_back(this->weight_table[1][tmpx]);            
                    }
                    else if (o_fn == SOFTMAX_TR) { 
                        tmpMAC->inbuffer.push_back(o_fn);
                        tmpMAC->inbuffer.push_back(in_x);
                        tmpMAC->inbuffer.push_back(causal_mask_flag); 
                        tmpMAC->inbuffer.insert(tmpMAC->inbuffer.end(), this->input_table[0].begin() + tmpy*in_x, this->input_table[0].begin() + tmpy*in_x + in_x); 
                    }
                    else if (o_fn == ADD) { 
                        tmpMAC->inbuffer.push_back(o_fn);
                        tmpMAC->inbuffer.push_back(in_x);
                        tmpMAC->inbuffer.insert(tmpMAC->inbuffer.end(), this->input_table[0].begin() + tmpy*in_x, this->input_table[0].begin() + tmpy*in_x + in_x); 
                        int residual_source_id = this->cnnmodel->all_layer_size[c_layer][1]; 
                        if (layer_outputs_history.find(residual_source_id) == layer_outputs_history.end()) {
                            residual_source_id = (c_layer > 0) ? (c_layer - 1) : 0;
                        }
                        if (layer_outputs_history[residual_source_id].empty()) {
                            tmpMAC->inbuffer.insert(tmpMAC->inbuffer.end(), in_x, 0.0); 
                        } else {
                            auto& residual_data = layer_outputs_history[residual_source_id][0];
                            int required_size = (tmpy * in_x) + in_x;
                            if (residual_data.size() < required_size) {
                                tmpMAC->inbuffer.insert(tmpMAC->inbuffer.end(), in_x, 0.0);
                            } else {
                                tmpMAC->inbuffer.insert(tmpMAC->inbuffer.end(), residual_data.begin() + tmpy*in_x, residual_data.begin() + tmpy*in_x + in_x);
                            }
                        }
                    }
                    else if (o_fn == EMBEDDING) { 
                        tmpMAC->inbuffer.push_back(o_fn);
                        tmpMAC->inbuffer.push_back(w_x);
                        int token_id = (int)this->input_table[0][tmpy]; 
                        assert(token_id >= 0 && token_id < (int)weight_table.size() && "Token ID out of vocab range");
                        tmpMAC->inbuffer.insert(tmpMAC->inbuffer.end(), this->weight_table[token_id].begin(), this->weight_table[token_id].end());
                    }
                    else if (o_fn == RMSNORM) { 
                        tmpMAC->inbuffer.push_back(o_fn);
                        tmpMAC->inbuffer.push_back(in_x);
                        float rms = this->precalc_rms[tmpy];
                        tmpMAC->inbuffer.push_back(rms);
                        tmpMAC->inbuffer.push_back(this->input_table[0][tmpy*in_x + tmpx]); 
                        tmpMAC->inbuffer.push_back(this->weight_table[0][tmpx]);            
                    }
                    else if (o_fn == SWIGLU || o_fn == GEGLU) { 
                        tmpMAC->inbuffer.push_back(o_fn);
                        tmpMAC->inbuffer.push_back(o_x); 
                        tmpMAC->inbuffer.push_back(this->input_table[0][tmpy*in_x + tmpx]);       
                        tmpMAC->inbuffer.push_back(this->input_table[0][tmpy*in_x + tmpx + o_x]); 
                    }
                    else if (o_fn == ROPE) { 
                        tmpMAC->inbuffer.push_back(o_fn);
                        tmpMAC->inbuffer.push_back(in_x);
                        tmpMAC->inbuffer.push_back(tmpy);
                        int q_dim = this->cnnmodel->all_layer_size[c_layer][1];
                        int n_heads = this->cnnmodel->all_layer_size[c_layer][3];
                        int head_dim = q_dim / n_heads; 
                        tmpMAC->inbuffer.push_back(head_dim);
                        int head_id = tmpx / head_dim;
                        int d = tmpx % head_dim;
                        int half_dim = head_dim / 2;
                        int pair_d = (d < half_dim) ? (d + half_dim) : (d - half_dim);
                        int pair_idx = (head_id * head_dim) + pair_d;
                        tmpMAC->inbuffer.push_back(this->input_table[0][tmpy*in_x + tmpx]);     
                        tmpMAC->inbuffer.push_back(this->input_table[0][tmpy*in_x + pair_idx]); 
                    }
                    else if (o_fn == ATTENTION) { 
                        int fused_dim = this->cnnmodel->all_layer_size[c_layer][0];
                        int q_dim =     this->cnnmodel->all_layer_size[c_layer][1];
                        int k_dim =     this->cnnmodel->all_layer_size[c_layer][2];
                        int n_heads =   this->cnnmodel->all_layer_size[c_layer][3];
                        
                        tmpMAC->inbuffer.push_back(o_fn);
                        tmpMAC->inbuffer.push_back(fused_dim);
                        tmpMAC->inbuffer.push_back(q_dim);
                        tmpMAC->inbuffer.push_back(k_dim);
                        tmpMAC->inbuffer.push_back(n_heads);
                        tmpMAC->inbuffer.push_back(tmpy); 
                        tmpMAC->inbuffer.push_back(tmpx); 
                        int limit = tmpy + 1; 
                        tmpMAC->inbuffer.insert(tmpMAC->inbuffer.end(), this->input_table[0].begin(), this->input_table[0].begin() + limit * fused_dim); 
                    }

                    int payload_size = tmpMAC->inbuffer.size();
#if ENABLE_KV_CACHE == 1
                    if (o_fn == ATTENTION) { 
                        int fused_dim = this->cnnmodel->all_layer_size[c_layer][0];
                        int k_dim = this->cnnmodel->all_layer_size[c_layer][2];
                        int kv_size_per_token = k_dim * 2; 
                        int tokens_in_cache = tmpMAC->local_sram_usage / kv_size_per_token;
                        bool is_autoregressive = (tmpy > 0);
                        
                        if (is_autoregressive && tokens_in_cache >= tmpy) {
                            int header_size = 7; 
                            payload_size = header_size + fused_dim; 
                            tmpMAC->inbuffer.clear();
                            tmpMAC->inbuffer.push_back(o_fn);
                            tmpMAC->inbuffer.push_back(fused_dim);
                            tmpMAC->inbuffer.push_back(this->cnnmodel->all_layer_size[c_layer][1]); 
                            tmpMAC->inbuffer.push_back(k_dim);
                            tmpMAC->inbuffer.push_back(this->cnnmodel->all_layer_size[c_layer][3]); 
                            tmpMAC->inbuffer.push_back(tmpy);
                            tmpMAC->inbuffer.push_back(tmpx);
                            tmpMAC->inbuffer.insert(tmpMAC->inbuffer.end(), this->input_table[0].begin() + tmpy * fused_dim, this->input_table[0].begin() + (tmpy + 1) * fused_dim);
                            int expected_sram = (tmpy + 1) * kv_size_per_token;
                            if (tmpMAC->local_sram_usage < expected_sram) {
                                if (expected_sram <= KV_CACHE_SIZE) tmpMAC->local_sram_usage = expected_sram;
                                else tmpMAC->local_sram_usage = KV_CACHE_SIZE; 
                            }
                        } else {
                            payload_size = tmpMAC->inbuffer.size();
                            int required_sram = (tmpy + 1) * kv_size_per_token;
                            if (required_sram <= KV_CACHE_SIZE) tmpMAC->local_sram_usage = required_sram;
                            else tmpMAC->local_sram_usage = KV_CACHE_SIZE; 
                        }
                    }
#endif
                    MAC_list[mem_id]->pecycle = cycles + ceil(payload_size * MEM_read_delay) + CACHE_DELAY;
                    MAC_list[mem_id]->inject(1, src, payload_size, o_fn, vcNetwork->NI_list[mem_id], pid, src_mac);
                }
            }
            it = tmpNI->packet_buffer_out[0].erase(it);
            Packet::release(tmpPacket);
        }

        for (auto it = tmpNI->packet_buffer_out[1].begin(); it != tmpNI->packet_buffer_out[1].end(); ) {
            tmpPacket = *it;
            if(tmpPacket->message.type != 2 || tmpPacket->message.out_cycle >= cycles)
            {
                ++it;
                continue;
            }
            src = tmpPacket->message.source_id;
            pid = tmpPacket->message.signal_id;
            src_mac = tmpPacket->message.mac_id;
            tmpMAC = MAC_list[src_mac];

#ifdef Countlatency
            if(pid*3+2 < CountNum) {
                DNN_latency[pid*3+2][0] = c_layer;
                DNN_latency[pid*3+2][1] = 2;
                DNN_latency[pid*3+2][2] = src_mac;
                DNN_latency[pid*3+2][4] = tmpPacket->send_out_time;
                DNN_latency[pid*3+2][7] = cycles;
            }
#endif

            if(this->cnnmodel->all_layer_type[c_layer]=='c'){ 
#ifdef newpooling
                if(this->cnnmodel->all_layer_type[c_layer+1]=='p') {
                    if(tmpPacket->message.data[0] >= this->output_table[tmpPacket->message.data[1]][tmpPacket->message.data[2]]) {
                        this->output_table[tmpPacket->message.data[1]][tmpPacket->message.data[2]] = tmpPacket->message.data[0]; 
                    }
                    if(tmpMAC->selfstatus == 5) tmpMAC->send = 3;
                } else {
#endif
#ifndef only3type
                if(tmpMAC->selfstatus == 4) {
                    if(tmpMAC->send == 1) {
                        this->output_table[tmpMAC->tmpch][tmpMAC->tmpm] = tmpMAC->outfeature;
                        MAC_list[mem_id]->inject(3,src,1,2,vcNetwork->NI_list[mem_id],pid, src_mac);
                    }
                }
#endif
#ifdef only3type
                this->output_table[tmpPacket->message.data[1]][tmpPacket->message.data[2]] = tmpPacket->message.data[0];
                if(tmpMAC->selfstatus == 5) tmpMAC->send = 3;
#endif
#ifdef newpooling
                }
#endif
            }
            else if(this->cnnmodel->all_layer_type[c_layer]=='p'){
#ifndef only3type
                if(tmpMAC->selfstatus == 4) {
                    if(tmpMAC->send == 1) {
                        this->output_table[tmpMAC->tmpch][tmpMAC->tmpm] = tmpMAC->outfeature;
                        MAC_list[mem_id]->inject(3,src,1,2,vcNetwork->NI_list[mem_id],pid, src_mac);
                    }
                }
#endif
#ifdef only3type
                this->output_table[tmpPacket->message.data[1]][tmpPacket->message.data[2]] = tmpPacket->message.data[0];
                if(tmpMAC->selfstatus == 5) tmpMAC->send = 3;
#endif
            }
            else if(this->cnnmodel->all_layer_type[c_layer]=='f'){ 
#ifndef only3type
                if(tmpMAC->selfstatus == 4) {
                    if(tmpMAC->send == 1) {
                        this->output_table[tmpMAC->tmpch][tmpMAC->tmpm] = tmpMAC->outfeature;
                        MAC_list[mem_id]->inject(3,src,1,2,vcNetwork->NI_list[mem_id],pid, src_mac);
                    }
                }
#endif
#ifdef only3type
                this->output_table[tmpPacket->message.data[1]][tmpPacket->message.data[2]] = tmpPacket->message.data[0];
                if(tmpMAC->selfstatus == 5) tmpMAC->send = 3;
#endif
            }
            else if(this->cnnmodel->all_layer_type[c_layer]=='m' || this->cnnmodel->all_layer_type[c_layer]=='l' || 
                     this->cnnmodel->all_layer_type[c_layer]=='s' || this->cnnmodel->all_layer_type[c_layer]=='a' || 
                     this->cnnmodel->all_layer_type[c_layer]=='e' || this->cnnmodel->all_layer_type[c_layer]=='r' || 
                     this->cnnmodel->all_layer_type[c_layer]=='w' || this->cnnmodel->all_layer_type[c_layer]=='o' || 
                     this->cnnmodel->all_layer_type[c_layer]=='t' || this->cnnmodel->all_layer_type[c_layer]=='g'){
#ifndef only3type
                if(tmpMAC->selfstatus == 4) {
                    if(tmpMAC->send == 1) {
                        this->output_table[tmpPacket->message.data[1]][tmpPacket->message.data[2]] = tmpPacket->message.data[0];
                        MAC_list[mem_id]->inject(3,src,1,2,vcNetwork->NI_list[mem_id],pid, src_mac);
                    }
                }
#endif
#ifdef only3type
                this->output_table[tmpPacket->message.data[1]][tmpPacket->message.data[2]] = tmpPacket->message.data[0];
                if(tmpMAC->selfstatus == 5) tmpMAC->send = 3;
#endif
            }
            it = tmpNI->packet_buffer_out[1].erase(it);
            Packet::release(tmpPacket);
        }
    }

    static std::vector<bool> is_dest(TOT_NUM, false);
    static bool init_dest = false;
    if (!init_dest) {
        for (int m = 0; m < MEM_NODES; m++) {
            if (dest_list[m] < TOT_NUM) {
                is_dest[dest_list[m]] = true;
            } else {
                std::cerr << "\n[FATAL ERROR] Node ID in dest_list (" << dest_list[m] 
                          << ") supera il limite della rete TOT_NUM (" << TOT_NUM << ")!\n";
                exit(EXIT_FAILURE);
            }
        }
        init_dest = true;
    }

    for(int i=0; i<TOT_NUM; i++){
        if (is_dest[i]) {continue;}

        tmpNI = this->vcNetwork->NI_list[i];
        for (auto it = tmpNI->packet_buffer_out[0].begin(); it != tmpNI->packet_buffer_out[0].end(); ) {
            tmpPacket = *it;
            
#ifdef cNoC_MODE
            if (tmpPacket->message.type == 4) {
                it = tmpNI->packet_buffer_out[0].erase(it);
                Packet::release(tmpPacket);
                continue;
            }
#endif

            if(tmpPacket->message.type != 1 || tmpPacket->message.out_cycle >= cycles)
            {
                ++it;
                continue;
            }
            src_mac = tmpPacket->message.mac_id; 
            pid = tmpPacket->message.signal_id;

#ifdef Countlatency
            if(pid*3 + 1 < CountNum) {
                DNN_latency[pid*3+1][4] = tmpPacket->send_out_time;
                DNN_latency[pid*3+1][7] = cycles;
            }
#endif
            tmpMAC = MAC_list[src_mac];
            tmpMAC->request = -1;
            it = tmpNI->packet_buffer_out[0].erase(it);
            Packet::release(tmpPacket);
        }

#ifndef only3type
        for (auto it = tmpNI->packet_buffer_out[1].begin(); it != tmpNI->packet_buffer_out[1].end(); ) {
            tmpPacket = *it;
            if(tmpPacket->message.type != 3)
            {
                ++it;
                continue;
            }
            src_mac = tmpPacket->message.mac_id;
            tmpMAC = MAC_list[src_mac];
            tmpMAC->send = 2;
            it = tmpNI->packet_buffer_out[1].erase(it);
            Packet::release(tmpPacket);
        }
#endif
    }
    return;
}

MACnet::~MACnet(){
    MAC* mac1;
    while (MAC_list.size()!=0){
        mac1 = MAC_list.back();
        MAC_list.pop_back();
        delete mac1;
    }
}