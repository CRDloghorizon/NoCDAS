/*
 * MACnet.cpp
 *
 */


#include "MACnet.hpp"

// helper function
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

	for(int i=0; i<macNum; i++){ // see ppt
		temp_ni_id = i%TOT_NUM;
		//cout << temp_ni_id << " " << i << endl;
	    MAC* nMAC = new MAC(i, this, temp_ni_id); // different from NN compute
	    MAC_list.push_back(nMAC);

	}
	// connect PEs, need update for NoC
	// for(int j=0; j < macNum - 1; j++){
	// 	MAC* self = MAC_list[j];
	// 	MAC* next = MAC_list[j+1];
	// 	self->nextMAC = next;
	// }

	// layer 0: input
	deque<int> layer_info;
	if(cnnmodel->all_layer_type[c_layer]!='i') // 0 layer
	{
		cout << "err: first layer is not input" << endl;
		//return 0;
	}
	layer_info = cnnmodel->all_layer_size[c_layer];
	in_x = layer_info[1]; // in_x
	in_y = layer_info[2]; // in_y
	in_ch = layer_info[3]; // in_ch
	c_layer++;

	// for new pooling
	no_x = 0;
	no_y = 0;
	nw_x = 0;
	nw_y = 0;
	no_ch = 0; // next o_ch in pooling layer
	npad = 0; // padding
	nstride = 1; // stride

	// layer 1: 1st conv layer
	layer_info = cnnmodel->all_layer_size[c_layer];
	if(cnnmodel->all_layer_type[c_layer]=='c')
	{
		w_x = layer_info[1]; // w_x
		w_y = layer_info[2]; // w_y
		o_ch = layer_info[3]; // o_ch
		w_ch = o_ch * in_ch; // w_ch = o_ch * in_ch
		o_fn = layer_info[5]; 
		pad = layer_info[6]; // padding
		stride = layer_info[7]; // stride
		//cout << layer_info[4] << ' ' << in_ch << ' ' << w_ch << endl;
		assert((in_ch == layer_info[4]) && "Input channel not correct!");
	}
	st_w = 0;
	// for 1st conv layer output
	o_x = (in_x + 2*pad - w_x) / stride + 1;
	o_y = (in_y + 2*pad - w_y) / stride + 1;
	readyflag = 0; //standby
	cout << "!!MACnet created!!" << endl;
	cout << "layer" << c_layer << " created " << cnnmodel->all_layer_type[c_layer] << ' ' << in_ch << ' ' << (o_ch * o_x * o_y) << endl;

	Layer_latency.clear();

}

void MACnet::create_input(){
	input_table.resize(in_ch);
	int outmatsize = o_x * o_y;
	int wmatsize = w_x * w_y;
	int padded_x = in_x + 2*pad;
	int padded_y = in_y + 2*pad;
	weight_table.resize(w_ch);

	// for input
	if (this->c_layer == 1) // 1st conv
	{
		// add padding with in_x, in_y
		for(int i=0;i<in_ch;i++){
			if(pad==0)
			{
				input_table[i].assign(this->cnnmodel->all_data_in[i].begin(),this->cnnmodel->all_data_in[i].end());
			}
			else{
				input_table[i].assign(padded_x * padded_y, 0.0);
				for(int p=0; p<in_y;++p)
				{
					for(int q=0;q<in_x;++q)
					{
						input_table[i][(p+pad)*padded_x + (q+pad)] = this->cnnmodel->all_data_in[i][p*in_x+q];
					}
				}
			}
		}		
	}
	else if (this->c_layer >=2)
	{
		for(int i=0;i<in_ch;i++){

			if(pad==0)
			{
				input_table[i].assign(this->output_table[i].begin(),this->output_table[i].end());
				//check1
				if (this->cnnmodel->all_layer_type[c_layer]=='f')
				{cout << "check1 " << i << " " << input_table[i].size() << " " << this->output_table[i].size() << endl;}
			}
			else{
				input_table[i].assign(padded_x * padded_y, 0.0);
				for(int p=0; p<in_y;++p)
				{
					for(int q=0;q<in_x;++q)
					{
						input_table[i][(p+pad)*padded_x + (q+pad)] = this->output_table[i][p*in_x+q];
					}
				}
			}
		}
	}

	// for weight
	if (this->cnnmodel->all_layer_type[c_layer]=='c')
	{
		//***************************
		for(int i=0;i<o_ch;i++){
			for(int j=0;j<in_ch;j++)
			{
				weight_table[i*in_ch + j].assign(this->cnnmodel->all_weight_in[st_w + i].begin() + j*wmatsize,this->cnnmodel->all_weight_in[st_w + i].begin() + j*wmatsize + wmatsize);	//weight filter
				weight_table[i*in_ch + j].push_back(this->cnnmodel->all_weight_in[st_w + i].back());	//bias
			}
		}
		st_w += o_ch;
	}
	else if (this->cnnmodel->all_layer_type[c_layer]=='f')
	{
		//check2
		cout << "check2 " << st_w << " " << w_ch << " " << this->cnnmodel->all_weight_in[st_w].size() << endl;
		for(int i=0;i<w_ch;i++){
			weight_table[i].assign(this->cnnmodel->all_weight_in[st_w + i].begin(),this->cnnmodel->all_weight_in[st_w + i].end());
		}
		st_w += w_ch;
		//check3
		cout << "check3 " << st_w << " " << weight_table[0].size() << " " << this->cnnmodel->all_weight_in[st_w-1].size() << endl;
	}
	else if (this->cnnmodel->all_layer_type[c_layer]=='p')
	{
		weight_table.clear();
	}


	// for output
#ifdef newpooling
	if(this->cnnmodel->all_layer_type[c_layer]=='c' && this->cnnmodel->all_layer_type[c_layer+1]=='p')
	{
		nw_x = cnnmodel->all_layer_size[c_layer+1][1];
		nw_y = cnnmodel->all_layer_size[c_layer+1][2];
		no_ch = cnnmodel->all_layer_size[c_layer+1][3]; // next o_ch in pooling layer
		npad = cnnmodel->all_layer_size[c_layer+1][4]; // padding
		nstride = cnnmodel->all_layer_size[c_layer+1][5]; // stride
		assert((o_ch == no_ch) && "Input channel not correct for merged pooling!");
		no_x = (o_x + 2*npad - nw_x) / nstride + 1;
		no_y = (o_y + 2*npad - nw_y) / nstride + 1;
		cout << cycles << " layer" << (c_layer+1) << " merged with pooling " << cnnmodel->all_layer_type[c_layer + 1] << ' ' << no_ch << ' ' << (no_ch * no_x * no_y) << endl;

		outmatsize = no_x * no_y;
//		pooling_table.resize(no_ch);
//		for(int i=0;i<o_ch;i++){
//			pooling_table[i].assign(outmatsize, 0.0);
//		}
	}
#endif
	output_table.resize(o_ch);
	for(int i=0;i<o_ch;i++){
		output_table[i].assign(outmatsize, 0.0);
	}

	return;
}

// default direct x mapping
void MACnet::mapping(int neuronnum){
	this->mapping_table.clear();
	this->mapping_table.resize(macNum);

	//dir_x, except dest_list

	int j = 0;
	while (j < neuronnum){
		for( int i = 0; i < macNum; i++)
		{
			int temp_i = i%TOT_NUM;
			if (contains(dest_list, temp_i)) {continue;}

			this->mapping_table[i].push_back(j);
			j = j + 1;
			if (j == neuronnum) break;
		}
	}
	return;
}

// default direct y mapping
void MACnet::ymapping(int neuronnum){
	this->mapping_table.clear();
	this->mapping_table.resize(macNum);

	int npos[macNum];
	for (int c = 0; c < PE_X_NUM; c++)
	{
		for (int r = 0; r < PE_Y_NUM; r++)
		{
			npos[c * PE_Y_NUM + r] = r * PE_X_NUM + c;
		}
	}

	//dir_y, col mapping, except dest_list
	int j = 0;
	int k = 0;
	while (j < neuronnum){
		for( int i = 0; i < macNum; i++)
		{
			k = npos[i];
			int temp_i = k%TOT_NUM;

			if (contains(dest_list, temp_i)) {continue;}

			this->mapping_table[k].push_back(j);
			j = j + 1;
			if (j == neuronnum) break;
		}
	}
	return;
}

// random mapping
void MACnet::rmapping(int neuronnum){
	this->mapping_table.clear();
	this->mapping_table.resize(macNum);

	unsigned seed = 0; // std::chrono::system_clock::now().time_since_epoch().count();
	vector<int> npos;
	for (int c=0; c<macNum; c++) {npos.push_back(c);}

	shuffle(npos.begin(), npos.end(), default_random_engine(seed));

	//random mapping, except dest_list
	int j = 0;
	int k = 0;
	while (j < neuronnum){
		for( int i = 0; i < macNum; i++)
		{
			k = npos[i];
			int temp_i = k%TOT_NUM;

			if (contains(dest_list, temp_i)) {continue;}

			this->mapping_table[k].push_back(j);
			j = j + 1;
			if (j == neuronnum) break;
		}
	}
	return;
}



void MACnet::checkStatus()
{
	if(readyflag == 0) // first conv layer
	{
		this->create_input();

#ifdef rowmapping
		this->mapping(o_ch * o_x * o_y);
#endif
#ifdef colmapping
		this->ymapping(o_ch * o_x * o_y);
#endif
#ifdef randmapping
		this->rmapping(o_ch * o_x * o_y);
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
				this->MAC_list[i]->routing_table.assign(mapping_table[i].begin(),mapping_table[i].end()); //mapping table
			}
		}

		readyflag = 1; // loading complete
		return;
	}
	
	// check running status in one layer
	for(int i=0; i<macNum; i++){
		if(MAC_list[i]->selfstatus != 5) {
			readyflag = 1;
			return;
		}
#ifdef only3type
		else
		{
			if(MAC_list[i]->send != 3)
			{
				//cout <<  MAC_list[i]->send << endl;
				readyflag = 1;
				return;
			}
		}
#endif
	}
	
	// after layer complete, fetch new layer
	deque<int> layer_info;
	in_x = o_x; // in_x
	in_y = o_y; // in_y
	in_ch = o_ch; // in_ch	
	
	#ifdef newpooling
		if(this->cnnmodel->all_layer_type[c_layer]=='c' && this->cnnmodel->all_layer_type[c_layer+1]=='p') //c+p is completed
		{
//			nw_x = cnnmodel->all_layer_size[c_layer+1][1];
//			nw_y = cnnmodel->all_layer_size[c_layer+1][2];
//			in_ch = cnnmodel->all_layer_size[c_layer+1][3]; // next o_ch in pooling layer
//			npad = cnnmodel->all_layer_size[c_layer+1][4]; // padding
//			nstride = cnnmodel->all_layer_size[c_layer+1][5]; // stride
//			in_x = (o_x + 2*npad - nw_x) / nstride + 1;
//			in_y = (o_y + 2*npad - nw_y) / nstride + 1;
			in_ch = no_ch;
			in_x = no_x;
			in_y = no_y;
			c_layer++;
		}
	#endif

	c_layer++; // go to next layer normally
	if(c_layer == n_layer)
	{
		cout << "All finished! at cycle " << cycles << endl;
		Layer_latency.push_back(cycles);
		cout << "Latency for all layers: " << endl ;

		for (auto element : Layer_latency) {
		        cout << element << endl;
		    }
		cout << endl;

		cout << "Latency for each layer: " << endl ;
		cout << Layer_latency[0] << endl;
		for (int lat = 0; lat < Layer_latency.size()-1; lat++) {
				cout << Layer_latency[lat+1] - Layer_latency[lat] << endl;
			}
		cout << endl;

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

	// reset rr_record in router inport and NI
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

	//fetch new layer
	layer_info = cnnmodel->all_layer_size[c_layer];
	if(cnnmodel->all_layer_type[c_layer]=='c')  // for conv layer output
	{
		w_x = layer_info[1]; // w_x
		w_y = layer_info[2]; // w_y
		o_ch = layer_info[3]; // o_ch
		w_ch = o_ch * in_ch; // w_ch = o_ch * in_ch
		o_fn = layer_info[5];  // 0 to 3
		pad = layer_info[6]; // padding
		stride = layer_info[7]; // stride

		assert((in_ch == layer_info[4]) && "Input channel not correct!");
		o_x = (in_x + 2*pad - w_x) / stride + 1;
		o_y = (in_y + 2*pad - w_y) / stride + 1;
		cout << cycles << " layer" << c_layer << " " << cnnmodel->all_layer_type[c_layer] << ' ' << in_ch << ' ' << (o_ch * o_x * o_y) << endl;
	}
	else if(cnnmodel->all_layer_type[c_layer]=='f')  // for fc layer output
	{
		// in_x = in_x * in_y * in_ch;
		in_x = layer_info[0];
		in_ch = 1;
		in_y = 1;
		w_x = layer_info[0]; // = in_x
		w_y = 1; // w_y
		o_ch = 1; // o_ch
		w_ch = layer_info[1]; // = o_x
		o_fn = layer_info[2] + 4; // 4 to 7
		pad = 0;
		stride = 1;
		assert((in_x == w_x) && "Input channel not correct!");
		o_x = layer_info[1];
		o_y = 1;
		cout << cycles << " layer" << c_layer << " " << cnnmodel->all_layer_type[c_layer] << ' ' << in_x << ' ' << w_x << ' ' << o_x << endl;
		if(this->output_table.size() > 1) //flatten
		{
			vector<float> temp_out_table;
			for(int z = 0; z < this->output_table.size(); z++)
			{
				temp_out_table.insert(temp_out_table.end(), this->output_table[z].begin(), this->output_table[z].end());
			}
			this->output_table.resize(1);
			this->output_table[0].assign(temp_out_table.begin(), temp_out_table.end());
			//cout << "tag 2 " << temp_out_table.size() << " " << this->output_table[0].size() << endl;
		}
	}
	else if(cnnmodel->all_layer_type[c_layer]=='p')  // for max pooling layer output
	{
		w_x = layer_info[1];
		w_y = layer_info[2];
		o_ch = layer_info[3]; // o_ch
		pad = layer_info[4]; // padding
		stride = layer_info[5]; // stride
		w_ch = 0;
		o_fn = 8; // max pooling
		if (layer_info[6] == 2) {o_fn = 12;} // average pooling
		assert((in_ch == o_ch) && "Input channel not correct!");
		o_x = (in_x + 2*pad - w_x) / stride + 1;
		o_y = (in_y + 2*pad - w_y) / stride + 1;
//		o_x = in_x / w_x;
//		o_y = in_y / w_y;
		cout << cycles << " layer" << c_layer << " " << cnnmodel->all_layer_type[c_layer] << ' ' << in_ch << ' ' << (o_ch * o_x * o_y) << endl;
	}
	readyflag = 0;
	// reset Mac status
	for(int i=0; i<macNum; i++){
		MAC_list[i]->selfstatus = 0;
		//added hard sync
		MAC_list[i]->pecycle = cycles;
	}	
	return;
}


void MACnet::runOneStep()
{
	MAC * tmpMAC;
	NI * tmpNI;
	Packet * tmpPacket;
	for(int i=0; i<macNum; i++){ // run one step for each MAC
		MAC_list[i]->runOneStep();
		//cout <<  "mac: " << i << ' ' << cycles << endl;
	}

	// check MEM, MEM id is from dest_list

	int pbuffersize;
	int src;
	int pid;
	int mem_id;
	int src_mac;
	for(int memidx=0;memidx<MEM_NODES;memidx++)
	{
		mem_id = dest_list[memidx];
		tmpNI = this->vcNetwork->NI_list[mem_id];
		// for message type 0 from MAC to MEM
		pbuffersize = tmpNI->packet_buffer_out[0].size();
		for (int j=0; j < pbuffersize; j++) {
			tmpPacket = tmpNI->packet_buffer_out[0].front();
			// added check if reached out cycle
			// check received packet at MEM from MAC type 0
			if(tmpPacket->message.type != 0 || tmpPacket->message.out_cycle >= cycles)
			{
				tmpNI->packet_buffer_out[0].pop_front();
				tmpNI->packet_buffer_out[0].push_back(tmpPacket);
				continue;
			}
			src = tmpPacket->message.source_id;
			pid = tmpPacket->message.signal_id;
			src_mac = tmpPacket->message.mac_id;

#ifdef Countlatency
			//statistics
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
			// cout << "MEM " << tmpPacket->message.destination << " receive type " << tmpPacket->message.type << " from MAC " << src << endl;
			tmpMAC = MAC_list[src_mac];
			if(this->cnnmodel->all_layer_type[c_layer]=='c'){ // conv layer fetch data
				if(tmpMAC->selfstatus == 2) // request data && this->cnnmodel->all_layer_type[c_layer]=='c'
				{
					tmpMAC->tmpch = tmpMAC->request / (o_x*o_y); //current output channel
					tmpMAC->tmpm  = tmpMAC->request % (o_x*o_y); //current output map id
					tmpMAC->npoolflag = 0;
					int tmpx = tmpMAC->tmpm % o_x;
					int tmpy = tmpMAC->tmpm / o_x;
					tmpMAC->inbuffer.clear();
					// inbuffer: [fn] [ch size] [map size] [i] [w + b]
					tmpMAC->inbuffer.push_back(o_fn);
					tmpMAC->inbuffer.push_back(in_ch);
					tmpMAC->inbuffer.push_back(w_x * w_y);
					// for conv input
					for (int k=0; k<in_ch; k++)
					{
						for (int p=0; p<w_y; p++)
						{
							tmpMAC->inbuffer.insert(tmpMAC->inbuffer.end(), this->input_table[k].begin() + (tmpy*stride+p)*(in_x+2*pad) + tmpx*stride, this->input_table[k].begin() + (tmpy*stride+p)*(in_x+2*pad) + tmpx*stride + w_x);
						}
					}
					// for conv weight
					for (int k=0; k<in_ch; k++)
					{
						tmpMAC->inbuffer.insert(tmpMAC->inbuffer.end(),this->weight_table[tmpMAC->tmpch*in_ch+k].begin(), this->weight_table[tmpMAC->tmpch*in_ch+k].end()-1); //weight
					}
					tmpMAC->inbuffer.push_back(this->weight_table[tmpMAC->tmpch*in_ch].back()); //bias
					// added new pooling
#ifdef newpooling
					if (this->cnnmodel->all_layer_type[c_layer+1]=='p')
					{
						tmpMAC->npoolflag = 1;
						// if no padding
						int n_tmpx;
						int n_tmpy;
						if(tmpx >= (no_x-1)*nstride+nw_x || tmpy >= (no_y-1)*nstride+nw_y) // outside pooling range
						{
							tmpMAC->n_tmpch = -1;
							tmpMAC->n_tmpm.clear();
							tmpMAC->inbuffer.assign(4, 10);
						}
						else //inside pooling range
						{
							tmpMAC->n_tmpch = tmpMAC->tmpch; //suppose channel is the same
							// loop the pooling output map
							for(n_tmpx=0; n_tmpx<no_x;n_tmpx++){
								for(n_tmpy=0; n_tmpy<no_y;n_tmpy++){
									if(tmpx >= n_tmpx*nstride && tmpx < n_tmpx*nstride + nw_x && tmpy >= n_tmpy*nstride && tmpy < n_tmpy*nstride + nw_y)
									{tmpMAC->n_tmpm.push_back(n_tmpx+n_tmpy*no_x);}
								}
							}
							//cout << tmpMAC->id << " " <<tmpMAC->n_tmpm.size() <<endl;
						}
					}
#endif

					// added send type 1
					MAC_list[mem_id]->pecycle = cycles + ceil((in_ch * w_x * w_y * 2 + 1) * MEM_read_delay)  + CACHE_DELAY;
					MAC_list[mem_id]->inject(1,src,tmpMAC->inbuffer.size(),o_fn,vcNetwork->NI_list[mem_id],pid,src_mac);

#ifdef Countlatency
					if(pid*3+1 < CountNum) {
						DNN_latency[pid*3+1][0] = (tmpMAC->inbuffer.size()*2 + 1)/FLIT_LENGTH + 1;
					}
#endif
					//tmpMAC->request = -1;
					//return;
				}
			}
			else if (this->cnnmodel->all_layer_type[c_layer]=='p') // pooling
			{
				if(tmpMAC->selfstatus == 2) // request data && this->cnnmodel->all_layer_type[c_layer]=='p'
				{
					tmpMAC->tmpch = tmpMAC->request / (o_x*o_y); //current output channel
					tmpMAC->tmpm  = tmpMAC->request % (o_x*o_y); //current output map id
					int tmpx = tmpMAC->tmpm % o_x;
					int tmpy = tmpMAC->tmpm / o_x;
					tmpMAC->inbuffer.clear();
					// inbuffer: [fn] [map size] [i]
					tmpMAC->inbuffer.push_back(o_fn); // 8 or 12 -> max or avg
					tmpMAC->inbuffer.push_back(w_x * w_y);
					for (int p=0; p<w_y; p++)
					{
						//tmpMAC->inbuffer.insert(tmpMAC->inbuffer.end(), this->input_table[tmpMAC->tmpch].begin() + (tmpy*w_y+p)*in_x + tmpx*w_x, this->input_table[tmpMAC->tmpch].begin() + (tmpy*w_y+p)*in_x + tmpx*w_x + w_x);
						tmpMAC->inbuffer.insert(tmpMAC->inbuffer.end(), this->input_table[tmpMAC->tmpch].begin() + (tmpy*stride+p)*(in_x+2*pad) + tmpx*stride, this->input_table[tmpMAC->tmpch].begin() + (tmpy*stride+p)*(in_x+2*pad) + tmpx*stride + w_x);
					}

					// added send type 1
					MAC_list[mem_id]->pecycle = cycles + ceil(w_x * w_y * MEM_read_delay) + CACHE_DELAY;
					MAC_list[mem_id]->inject(1,src,tmpMAC->inbuffer.size(),o_fn,vcNetwork->NI_list[mem_id],pid,src_mac);

					//tmpMAC->request = -1;
					//return;
				}
			}
			else if (this->cnnmodel->all_layer_type[c_layer]=='f'){ // fc layer
				if(tmpMAC->selfstatus == 2) // request data && this->cnnmodel->all_layer_type[c_layer]=='f'
				{
					tmpMAC->tmpch = 0; //current output channel 1*ox*1
					tmpMAC->tmpm  = tmpMAC->request; //current output vector id (also w_ch)
					tmpMAC->npoolflag = 0;
					tmpMAC->inbuffer.clear();
					// inbuffer: [fn] [map size] [i] [w + b]
					tmpMAC->inbuffer.push_back(o_fn);
					tmpMAC->inbuffer.push_back(w_x * w_y);

					tmpMAC->inbuffer.insert(tmpMAC->inbuffer.end(), this->input_table[0].begin(), this->input_table[0].end());

					tmpMAC->inbuffer.insert(tmpMAC->inbuffer.end(),this->weight_table[tmpMAC->tmpm].begin(), this->weight_table[tmpMAC->tmpm].end()); //weight

					//DNN_latency[pid][3] = cycles;
					// added send type 1
					MAC_list[mem_id]->pecycle = cycles + ceil((w_x * w_y * 2 + 1) * MEM_read_delay) + CACHE_DELAY;
					MAC_list[mem_id]->inject(1,src,tmpMAC->inbuffer.size(),o_fn,vcNetwork->NI_list[mem_id],pid,src_mac);
					//DNN_latency[pid][4] = cycles;
					//tmpMAC->request = -1;
					//return;
				}
			}
			tmpNI->packet_buffer_out[0].pop_front();
		}

		// for message type 2 from MAC to MEM, received OFmap, MEM write
		pbuffersize = tmpNI->packet_buffer_out[1].size();
		for (int j=0; j < pbuffersize; j++) {
			tmpPacket = tmpNI->packet_buffer_out[1].front();
			 if(tmpPacket->message.type != 2 || tmpPacket->message.out_cycle >= cycles)
			{
				tmpNI->packet_buffer_out[1].pop_front();
				tmpNI->packet_buffer_out[1].push_back(tmpPacket);
				// cout << "continue macnet: " << cycles << endl;
				continue;
			}
			src = tmpPacket->message.source_id;
			pid = tmpPacket->message.signal_id;
			src_mac = tmpPacket->message.mac_id;
			// cout << "MEM " << tmpPacket->message.destination <<  " receive type " << tmpPacket->message.type << " from MAC " << src << endl;
			tmpMAC = MAC_list[src_mac];

#ifdef Countlatency
			//statistics
			if(pid*3+2 < CountNum) {
				DNN_latency[pid*3+2][0] = c_layer;
				DNN_latency[pid*3+2][1] = 2;
				DNN_latency[pid*3+2][2] = src_mac;
				DNN_latency[pid*3+2][4] = tmpPacket->send_out_time;
				DNN_latency[pid*3+2][7] = cycles;
			}
#endif

			if(this->cnnmodel->all_layer_type[c_layer]=='c'){ // conv
#ifdef newpooling
				if(this->cnnmodel->all_layer_type[c_layer+1]=='p')
				{
					if(tmpPacket->message.data[0] >= this->output_table[tmpPacket->message.data[1]][tmpPacket->message.data[2]]) // n_tmpch and n_tmpm
					{
						this->output_table[tmpPacket->message.data[1]][tmpPacket->message.data[2]] = tmpPacket->message.data[0]; // for max pooling
					}
					if(tmpMAC->selfstatus == 5) tmpMAC->send = 3;
				}
				else
				{
#endif

#ifndef only3type
				if(tmpMAC->selfstatus == 4) // receive data
				{
					if(tmpMAC->send == 1)
					{
						this->output_table[tmpMAC->tmpch][tmpMAC->tmpm] = tmpMAC->outfeature;
						// added send type 3
						MAC_list[mem_id]->inject(3,src,1,2,vcNetwork->NI_list[mem_id],pid, src_mac);
					}
				}
#endif
#ifdef only3type
				// new added
				this->output_table[tmpPacket->message.data[1]][tmpPacket->message.data[2]] = tmpPacket->message.data[0];
				if(tmpMAC->selfstatus == 5) tmpMAC->send = 3;
				//tmpMAC->send = 2;
#endif
#ifdef newpooling
				}
#endif
			}

			else if(this->cnnmodel->all_layer_type[c_layer]=='p'){
#ifndef only3type
				if(tmpMAC->selfstatus == 4) // receive data
				{
				if(tmpMAC->send == 1)
					{
						this->output_table[tmpMAC->tmpch][tmpMAC->tmpm] = tmpMAC->outfeature;

						// added send type 3
						MAC_list[mem_id]->inject(3,src,1,2,vcNetwork->NI_list[mem_id],pid, src_mac);
					}
				}
#endif
#ifdef only3type
			// new added
				this->output_table[tmpPacket->message.data[1]][tmpPacket->message.data[2]] = tmpPacket->message.data[0];
				//tmpMAC->send = 2;
				if(tmpMAC->selfstatus == 5) tmpMAC->send = 3;
#endif
			}
			else if(this->cnnmodel->all_layer_type[c_layer]=='f'){ // fc layer

#ifndef only3type
				if(tmpMAC->selfstatus == 4) // receive data
				{
					if(tmpMAC->send == 1)
					{
						this->output_table[tmpMAC->tmpch][tmpMAC->tmpm] = tmpMAC->outfeature;
						// added send type 3
						MAC_list[mem_id]->inject(3,src,1,2,vcNetwork->NI_list[mem_id],pid, src_mac);
					}
				}
#endif
#ifdef only3type
				// new added
				this->output_table[tmpPacket->message.data[1]][tmpPacket->message.data[2]] = tmpPacket->message.data[0];
				//tmpMAC->send = 2;
				if(tmpMAC->selfstatus == 5) tmpMAC->send = 3;
#endif
			}
			tmpNI->packet_buffer_out[1].pop_front();
		}
	}

	// only check non-mem node
	for(int i=0; i<TOT_NUM; i++){
		// skip mem nodes
		if (contains(dest_list, i)) {continue;}

		tmpNI = this->vcNetwork->NI_list[i];
		// for message type 1 from MEM to MAC
		pbuffersize = tmpNI->packet_buffer_out[0].size();
		for (int j=0; j < pbuffersize; j++) {
			tmpPacket = tmpNI->packet_buffer_out[0].front();
			if(tmpPacket->message.type != 1 || tmpPacket->message.out_cycle >= cycles)
			{
				tmpNI->packet_buffer_out[0].pop_front();
				tmpNI->packet_buffer_out[0].push_back(tmpPacket);
				continue;
			}
			src_mac = tmpPacket->message.mac_id; //mac
			pid = tmpPacket->message.signal_id;

#ifdef Countlatency
			if(pid*3 + 1 < CountNum) {
				DNN_latency[pid*3+1][4] = tmpPacket->send_out_time;
				DNN_latency[pid*3+1][7] = cycles;
			}
#endif
			//cout << cycles <<" MAC " << src_mac <<  " receive type " << tmpPacket->message.type << " from MEM " << tmpPacket->message.source_id << endl;
			tmpMAC = MAC_list[src_mac];
			tmpMAC->request = -1;
			tmpNI->packet_buffer_out[0].pop_front();

		}
		//changed to reduce trans delay
		// for message type 3 from MEM to MAC
#ifndef only3type
		pbuffersize = tmpNI->packet_buffer_out[1].size();
		for (int j=0; j < pbuffersize; j++) {
			tmpPacket = tmpNI->packet_buffer_out[1].front();
			if(tmpPacket->message.type != 3)
			{
				tmpNI->packet_buffer_out[1].pop_front();
				tmpNI->packet_buffer_out[1].push_back(tmpPacket);
				continue;
			}
			src_mac = tmpPacket->message.mac_id;

			//cout << cycles <<" MAC " << src_mac <<  " receive type " << tmpPacket->message.type << " from MEM " << tmpPacket->message.source_id << endl;
			tmpMAC = MAC_list[src_mac];
			tmpMAC->send = 2;
			tmpNI->packet_buffer_out[1].pop_front();
		}
#endif
	}

	return;
}

// Destructor
MACnet::~MACnet (){
	MAC* mac1;
	while (MAC_list.size()!=0){
		mac1 = MAC_list.back();
		MAC_list.pop_back();
	    delete mac1;
	}
}
