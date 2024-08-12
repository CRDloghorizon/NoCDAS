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

	// find dest id
	//dest_mem_id = 5;
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
#elif defined MemNode18 // 12*12
	if (xid <= 3) {
		dest_mem_id = dest_list[(yid/2)];
	} else if (xid <= 7 && xid > 3 ) {
		dest_mem_id = dest_list[(yid/2) + 6];
	} else {
		dest_mem_id = dest_list[(yid/2) + 12];
	}
#elif defined MemNode32 // 16*16
	if (xid <= 3) {
		dest_mem_id = dest_list[(yid/2)];
	} else if (xid <= 7 && xid > 3 ) {
		dest_mem_id = dest_list[(yid/2) + 8];
	} else if (xid <= 11 && xid > 7 ) {
		dest_mem_id = dest_list[(yid/2) + 16];
	} else {
		dest_mem_id = dest_list[(yid/2) + 24];
	}
#elif defined MemNode5 // 6*6
	if (xid <= 3) {
		dest_mem_id = dest_list[(yid/2)];
	} else {
		dest_mem_id = dest_list[(yid/3) + 3];
	}
#elif defined MemNode13 // 10*10
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
//	int x_id, y_id, d_x_id, d_y_id;
//	x_id = id / X_NUM;
//	y_id = id % X_NUM;
//	d_x_id = d_id / X_NUM;
//	d_y_id = d_id % X_NUM;
	Message msg;
	msg.NI_id = NI_id;
	msg.mac_id = mac_src; //MAC
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
  	msg.slave_id = d_id; //NI
  	msg.source_id = NI_id; // NI
  	msg.type = type; // 0 1 2 3

	Packet* packet = new Packet(msg, X_NUM, t_NI->NI_num);
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
			request = routing_table.front();
			tmp_request = request;
			routing_table.pop_front();
			//send_request(), fill inbuffer type 0
			inject(0, dest_mem_id, 1, request, net->vcNetwork->NI_list[NI_id], packet_id + request, id);
			selfstatus = 2;
			pecycle = cycles;
#ifdef Countlatency
			//statistics
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
					this->selfstatus = 0; 				// back to initial state
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
			if (fn >=0 && fn <=3){ // Conv [fn] [ch size] [map size] [i] [w + b]
				ch_size = inbuffer[1];
				m_size = inbuffer[2];
				infeature.assign(inbuffer.begin() + 3, inbuffer.begin() + 3 + ch_size * m_size); //input
				weight.assign(inbuffer.begin() + 3 + ch_size * m_size, inbuffer.end()); // w matrix + b (ch_size * m_size + 1)
				assert((weight.size() == ch_size * m_size + 1) && "Weight not correct after request (Conv)");
			}
			else if (fn >= 4 && fn <= 7) // fc [fn] [map size] [i] [w + b]
			{
				ch_size = 1;
				m_size = inbuffer[1];
				infeature.assign(inbuffer.begin() + 2, inbuffer.begin() + 2 + m_size);
				weight.assign(inbuffer.begin() + 2 + m_size, inbuffer.end()); //w + b
			}
			else if (fn == 8) // max pooling [fn] [map size] [i]
			{
				ch_size = 1;
				m_size = inbuffer[1];
				infeature.assign(inbuffer.begin() + 2, inbuffer.end());
				assert((infeature.size() == m_size) && "Inbuffer not correct after request (pooling)");
			}
			outfeature = 0.0;
			selfstatus = 3;
			pecycle = cycles;
			return;
		}
		else if(selfstatus == 3){
			// normal MAC op
			if (fn >=0 && fn <=3) // Conv
			{
				for(int i=0; i < ch_size; i++)
				{
					for(int j=0; j < m_size; j++)
					{
						//outfeature += infeature[i*m_size + j] * weight[i*(m_size+1) + j];
						outfeature += infeature[i*m_size + j] * weight[i*m_size + j];
					}
					//outfeature += weight[i*m_size + m_size];
				}
				outfeature += weight[ch_size*m_size]; //bias only added once per output channel
			}
			else if (fn >= 4 && fn <= 7) // FC
			{
				for(int j=0; j < m_size; j++)
				{
					outfeature += infeature[j] * weight[j];
				}
				outfeature += weight[m_size];
			}
			else if (fn == 8) // max pooling
			{
				outfeature = infeature[0];
				for(int j=1; j < m_size; j++)
				{
					if (infeature[j] > outfeature) {outfeature = infeature[j];}
				}
				selfstatus = 4; // ready for this computation
				pecycle = cycles + 1; // sync cycles

				inject(2, dest_mem_id, 1, outfeature, net->vcNetwork->NI_list[NI_id], packet_id + tmp_request, id);
#ifdef Countlatency
				//statistics
				stats1 = (packet_id + tmp_request)*3 + 2;
				if(stats1 < CountNum) {
					DNN_latency[stats1][3] = pecycle;
				}
#endif
				//packet_id++;
				return;
			}

			int calctime = (ch_size * m_size / PE_NUM_OP + 1) * PE_FREQ_RATIO;  //25, 10
			//int calctime = 25;
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
					this->selfstatus = 0; 				// back to initial state
				}
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
				this->selfstatus = 0; 				// back to initial state
			}
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
