/*
 * NI.cpp
 *
 */



#include "NI.hpp"
#include "../parameters.hpp"

int NI::count_s=0;
int NI::count_r=0;
int NI::count_s_resp=0;
int NI::count_r_resp=0;
int NI::count_input=0;

int NI::LCS_delay_distribution[DISTRIBUTION_NUM]={0};
int NI::URS_delay_distribution[DISTRIBUTION_NUM]={0};

int NI::worst_LCS = 0;
int NI::worst_URS = 0;



int NI::URS_latency_single_dis[100] = {0};
int NI::LCS_latency_single_dis[100] = {0};
int NI::converse_latency_single_dis[100] = {0};



int NI::URS_latency_single_count = 0;
int NI::URS_latency_single_total = 0;
int NI::URS_latency_single_worst = 0;


int NI::LCS_latency_single_count = 0;
int NI::LCS_latency_single_total = 0;
int NI::LCS_latency_single_worst = 0;

int NI::converse_latency_single_count = 0;
int NI::converse_latency_single_total = 0;
int NI::converse_latency_single_worst = 0;



NI::NI (int t_id, VCNetwork* t_vcNetwork, int t_vn_num, int t_vc_per_vn, int t_vc_priority_per_vn, int t_in_depth, int* t_NI_num)
{
  id = t_id;
  vcNetwork = t_vcNetwork;
  vn_num = t_vn_num;
  vc_per_vn = t_vc_per_vn;
  vc_priority_per_vn = t_vc_priority_per_vn;
  NI_num = t_NI_num;
  // output;
  packetBuffer_list.reserve(vn_num);
  for(int i=0; i<vn_num; i++){
      PacketBuffer* t_packetBuffer = new PacketBuffer(this, i);
      packetBuffer_list.push_back(t_packetBuffer);
  }
  buffer_list.reserve(vn_num*(vc_per_vn+vc_priority_per_vn));

  for(int i=0; i<vn_num*vc_per_vn; i++){
      FlitBuffer * t_flitBuffer = new FlitBuffer(i%vc_per_vn, i/vc_per_vn, i, INFINITE1);
      buffer_list.push_back(t_flitBuffer);
      out_vc.push_back(-1);
      state.push_back(0);
  }

  for(int i=0; i<vn_num*vc_priority_per_vn; i++){
      FlitBuffer * t_flitBuffer = new FlitBuffer(i%vc_priority_per_vn+vc_per_vn, i/vc_priority_per_vn, i, INFINITE1);
      buffer_list.push_back(t_flitBuffer);
      out_vc.push_back(-1);
      state.push_back(0);
  }

  rr_buffer = 0;
  in_depth = t_in_depth;

  // output;
  ROutPort * vc_rOutPort = new ROutPort(0, 1, 1, 0, INFINITE);
  out_port = vc_rOutPort;

  // input;
  Link * link = new Link((RInPort*)NULL);
  RInPort * vcInPort = new RInPort(0, vn_num, vc_per_vn, vc_priority_per_vn, INFINITE, this, link);
  vcInPort->rid[0] = t_id/X_NUM;
  vcInPort->rid[1] = t_id%X_NUM;
  link->rInPort = vcInPort;
  in_port = vcInPort;

  count_vc = 0;
  count_switch = 0;
  packet_buffer_out.resize(2);

  starvation = 0;
  rr_priority_record = 0;

  num_flit = 0;

  total_delay = 0;
  total_num = 0;
  total_delay_URS = 0;
  total_num_URS = 0;

  // for new pooling
  NI_cycle_npooling = 0;
}


bool NI::flitize(Packet* packet, int vn){
  int base;
  //added to replace global packet_id
  int pid = packet->message.signal_id;
  // for priority packet with individual VCs
  if(packet->message.QoS == 3){
      base = vn_num*vc_per_vn+vn*vc_priority_per_vn;
      for(int i=base; i<base+vc_priority_per_vn; i++){
	  if(state[i] == 0){  // idle
	      //packet_id++;  // if success
	      int length = (packet->length-1)/FLIT_LENGTH+1; //round up
	      packet->send_out_time = cycles;
	      num_flit += length;
	      if(length == 1){
		  Flit* flit = new Flit(0, 10, vn, i, packet, cycles, pid);
		  flit->trace_node.push_back(NI::id);
		  flit->trace_time.push_back(cycles);
		  flit->packet->in_net_time = cycles;
		  buffer_list[i]->enqueue(flit);
	      }else{
		  for(int id=0; id<length; id++){
		      if(id == 0){
			  Flit* flit = new Flit(id, 0, vn, i, packet, cycles, pid);
			  flit->trace_node.push_back(NI::id);
			  flit->trace_time.push_back(cycles);
			  flit->packet->in_net_time = cycles;
			  buffer_list[i]->enqueue(flit);
		      }else
			if(id == length-1){
			    Flit* flit = new Flit(id, 1, vn, i, packet, cycles, pid);
			    flit->trace_node.push_back(NI::id);
			    flit->trace_time.push_back(cycles);
			    flit->packet->in_net_time = cycles;
			    buffer_list[i]->enqueue(flit);
			}else{
			    Flit* flit = new Flit(id, 2, vn, i, packet, cycles, pid);
			    flit->trace_node.push_back(NI::id);
			    flit->trace_time.push_back(cycles);
			    flit->packet->in_net_time = cycles;
			    buffer_list[i]->enqueue(flit);
			}
		  }
	      }
	      state[i] = 1;  // wait for VC allocation
	      return true;
	  }
      }
      return false;
  }

  // for normal packet or priority packet with shared VCs
  else{
#ifdef SHARED_PRI
      if(packet->message.QoS == 1){ // for LCS to use all the VCs
#endif

      base = vn*vc_per_vn; // 0 or 2  -> buffer_list(0,1) or buffer_list(2,3)
      for(int i=base; i<base+vc_per_vn; i++){
	  if(state[i] == 0){  // idle
	      //packet_id++;  // if success
	      packet->send_out_time = cycles;
	      int length = (packet->length-1)/FLIT_LENGTH + 1; //round up
	      num_flit += length;
	      if(length == 1){
		  Flit* flit = new Flit(0, 10, vn, i, packet, cycles, pid);
		  flit->trace_node.push_back(NI::id);
		  flit->trace_time.push_back(cycles);
		  flit->packet->in_net_time = cycles;
		  buffer_list[i]->enqueue(flit);
		  if(flit->packet->message.QoS == 1){
		      priority_vc.push_back(i);
		      priority_switch.push_back(i);
		  }
	      }else{
		  for(int id=0; id<length; id++){
		      if(id == 0){
			  Flit* flit = new Flit(id, 0, vn, i, packet, cycles, pid);
			  flit->trace_node.push_back(NI::id);
			  flit->trace_time.push_back(cycles);
			  flit->packet->in_net_time = cycles;
			  buffer_list[i]->enqueue(flit);
			  if(flit->packet->message.QoS == 1){
			      priority_vc.push_back(i);
			      priority_switch.push_back(i);
			  }
		      }else
			if(id == length-1){
			    Flit* flit = new Flit(id, 1, vn, i, packet, cycles, pid);
			    flit->trace_node.push_back(NI::id);
			    flit->trace_time.push_back(cycles);
			    flit->packet->in_net_time = cycles;
			    buffer_list[i]->enqueue(flit);
			}else{
			    Flit* flit = new Flit(id, 2, vn, i, packet, cycles, pid);
			    flit->trace_node.push_back(NI::id);
			    flit->trace_time.push_back(cycles);
			    flit->packet->in_net_time = cycles;
			    buffer_list[i]->enqueue(flit);
			}
		  }
	      }
	      state[i] = 1;  // wait for VC allocation
	      return true;
	  }
      }

#ifdef SHARED_PRI
      }
      else{  // for URS to use only vc_per_vn without priority (vc_priority_per_vn)
	  base = vn*vc_per_vn;
	  for(int i=base; i<base+vc_per_vn-vc_priority_per_vn; i++){
	    if(state[i] == 0){  // idle
		//packet_id++;  // if success
		packet->send_out_time = cycles;
		int length = (packet->length-1)/FLIT_LENGTH+1; //round up
		num_flit += length;
		if(length == 1){
		    Flit* flit = new Flit(0, 10, vn, i, packet, cycles, pid);
		    buffer_list[i]->enqueue(flit);
		}else{
		    for(int id=0; id<length; id++){
			if(id == 0){
			    Flit* flit = new Flit(id, 0, vn, i, packet, cycles, pid);
			    buffer_list[i]->enqueue(flit);
			}else
			  if(id == length-1){
			      Flit* flit = new Flit(id, 1, vn, i, packet, cycles, pid);
			      buffer_list[i]->enqueue(flit);
			  }else{
			      Flit* flit = new Flit(id, 2, vn, i, packet, cycles, pid);
			      buffer_list[i]->enqueue(flit);
			  }
		    }
		}
		state[i] = 1;  // wait for VC allocation
		return true;
	    }
	  }
      }
#endif
      return false;


  }
}

//
void NI::packetDequeue(){
  for(int vn=0; vn<vn_num; vn++){
      while(packetBuffer_list[vn]->packet_num != 0 && packetBuffer_list[vn]->read()->message.out_cycle < cycles){

	  Packet* packet = packetBuffer_list[vn]->read();
	  assert(vn == packet->vnet);

	  if(flitize(packet, vn)){
	      packetBuffer_list[vn]->dequeue();
	      if(packet->type == 0){
		      count_s++;
			  // cout << "NI send " << count_s << endl;
	      }
	      if(packet->type == 1){
		      count_s_resp++;
			  // cout << "NI send response " << count_s_resp << endl;
	      }
	  }
	  else{
	      break;
	  }

      }
  }
}

void NI::vcAllocation(){
  //cout << "#######" << endl;
  // for priority packet (shared VCs) QoS = 1

#ifdef SHARED_VC
  std::vector<int>::iterator iter;
  for(iter=priority_vc.begin(); iter<priority_vc.end();){
      if(count_vc == STARVATION_LIMIT)
	break;
      int tag = (*iter);
      if(state[tag]==1){
      	  int vc = out_port->out_link->rInPort->vc_allocate(buffer_list[tag]->read());
      	  if (vc != -1){
      	      out_vc[tag] = vc;
      	      state[tag] = 2; // active
      	      count_vc++;
      	      iter = priority_vc.erase(iter);
      	  }else
      	    iter++;
      }else
	iter++;
  }
  count_vc = 0;
#endif

  // for priority packet (individual VCs) QoS = 3
  for(int i=vn_num*vc_per_vn; i<vn_num*(vc_per_vn+vc_priority_per_vn); i++){
      int tag = (i+rr_priority_record)%(vn_num*vc_priority_per_vn)+vn_num*vc_per_vn;
      if(state[tag]==1){
	  int vn_rank = (tag-vn_num*vc_per_vn)/vc_priority_per_vn;
  	  int vc = out_port->out_link->rInPort->vc_allocate_priority(vn_rank);  // 2+vn_rank+0 (individual)
  	  if (vc != -1){
  	      out_vc[tag] = vc;
  	      state[tag] = 2; // active
  	  }
      }
  }

  // for normal packet QoS = 0
  for(int i=0; i<vn_num*vc_per_vn; i++){
      int tag = (i+rr_buffer)%(vn_num*vc_per_vn);
      if(state[tag]==1){

#ifdef SHARED_PRI
	  int vc = out_port->out_link->rInPort->vc_allocate_normal(buffer_list[tag]->read());
#else
	  int vc = out_port->out_link->rInPort->vc_allocate(buffer_list[tag]->read());
#endif

	  if (vc != -1){
	      out_vc[tag] = vc;
	      state[tag] = 2; // active
	  }
      }
  }

 // in case of no normal packet
#ifdef SHARED_VC
  for(;iter<priority_vc.end();){
      int tag = (*iter);
      if(state[tag]==1){
      	  int vc = out_port->out_link->rInPort->vc_allocate(buffer_list[tag]->read());
      	  if (vc != -1){
      	      out_vc[tag] = vc;
      	      state[tag] = 2; // active
      	      iter = priority_vc.erase(iter);
      	  }else
      	    iter++;
      }else
	iter++;
  }
#endif
}

void NI::switchArbitration(){

  // for priority packet (shared VCs)

#ifdef SHARED_VC
  std::vector<int>::iterator iter;
  for(iter=priority_switch.begin(); iter<priority_switch.end();iter++){
      if(count_switch == STARVATION_LIMIT)
	break;
      int tag = (*iter);
      if(buffer_list[tag]->cur_flit_num!=0){
	  Flit* flit = buffer_list[tag]->read();

	  if(state[tag]==2 && flit->sched_time < cycles && !out_port->out_link->rInPort->buffer_list[out_vc[tag]]->isFull()){
	      buffer_list[tag]->dequeue();
	      flit->vc = out_vc[tag];

	     // if(out_port->buffer_list[0]->isFull())
	     	    // cout<< "ni switch" << endl;
	      // added
	      flit->trace_node.push_back(NI::id);
	      flit->trace_time.push_back(cycles);
	      out_port->buffer_list[0]->enqueue(flit);
	      out_port->out_link->rInPort->buffer_list[out_vc[tag]]->get_credit();
	      flit->sched_time = cycles;
	      count_switch++;
	      if(flit->type == 1 || flit->type == 10){
		  state[tag] = 0; //idle
		  priority_switch.erase(iter);
	      }
	      return;
	  }
      }
  }
  count_switch = 0;
#endif


  // for priority packet (individual VCs)
  for(int i=vn_num*vc_per_vn; i<vn_num*(vc_per_vn+vc_priority_per_vn); i++){
      if(starvation == STARVATION_LIMIT){
	  //cout << starvation << ' ' << id << endl;
	  starvation = 0;
          break;
      }
      int tag = (i+rr_priority_record)%(vn_num*vc_priority_per_vn)+vn_num*vc_per_vn;
      if(buffer_list[tag]->cur_flit_num!=0){
  	  Flit* flit = buffer_list[tag]->read();
  	  if(state[tag]==2 && flit->sched_time < cycles && !out_port->out_link->rInPort->buffer_list[out_vc[tag]]->isFull()){
  	      buffer_list[tag]->dequeue();
  	      flit->vc = out_vc[tag];
  	      // added
//	      flit->trace_node.push_back(NI::id);
//	      flit->trace_time.push_back(cycles);
  	      out_port->buffer_list[0]->enqueue(flit);
  	      out_port->out_link->rInPort->buffer_list[out_vc[tag]]->get_credit();
  	      flit->sched_time = cycles;
  	      if(flit->type == 1 || flit->type == 10)
  		state[tag] = 0; //idle
  	      rr_priority_record = (rr_priority_record+1)%(vn_num*vc_priority_per_vn);
  	      starvation++;
  	      return;
          }
      }
  }


  // for normal packet
  for(int i=0; i<vn_num*vc_per_vn; i++){
      int tag = (i+rr_buffer)%(vn_num*vc_per_vn);
      if(buffer_list[tag]->cur_flit_num!=0){
	      Flit* flit = buffer_list[tag]->read();
	      if(state[tag]==2 && flit->sched_time < cycles && !out_port->out_link->rInPort->buffer_list[out_vc[tag]]->isFull()){
		  buffer_list[tag]->dequeue();
		  flit->vc = out_vc[tag];
		  // added
//		  flit->trace_node.push_back(NI::id);
//		  flit->trace_time.push_back(cycles);
		  out_port->buffer_list[0]->enqueue(flit);
		  out_port->out_link->rInPort->buffer_list[out_vc[tag]]->get_credit();
		  flit->sched_time = cycles;
		  if(flit->type == 1 || flit->type == 10)
		    state[tag] = 0; //idle
		  rr_buffer = (1+rr_buffer)%(vn_num*vc_per_vn);  // rr + 1, the next buffer will have the priority to switch.
		  return;
	      }
      }
  }

  // in case there is no normal packet

#ifdef SHARED_VC
  for(; iter<priority_switch.end();iter++){
      int tag = (*iter);
      if(buffer_list[tag]->cur_flit_num!=0){
	  Flit* flit = buffer_list[tag]->read();
	  if(state[tag]==2 && flit->sched_time < cycles && !out_port->out_link->rInPort->buffer_list[out_vc[tag]]->isFull()){
	      buffer_list[tag]->dequeue();
	      flit->vc = out_vc[tag];

	     // if(out_port->buffer_list[0]->isFull())
	     	    // cout<< "ni switch" << endl;
	      // added
	      flit->trace_node.push_back(NI::id);
	      flit->trace_time.push_back(cycles);
	      out_port->buffer_list[0]->enqueue(flit);
	      out_port->out_link->rInPort->buffer_list[out_vc[tag]]->get_credit();
	      flit->sched_time = cycles;
	      if(flit->type == 1 || flit->type == 10){
		  state[tag] = 0; //idle
		  priority_switch.erase(iter);
	      }
	      return;
	  }
      }
  }
#endif


  // for priority packet (individual VCs)
  for(int i=vn_num*vc_per_vn; i<vn_num*(vc_per_vn+vc_priority_per_vn); i++){
      int tag = (i+rr_priority_record)%(vn_num*vc_priority_per_vn)+vn_num*vc_per_vn;
      if(buffer_list[tag]->cur_flit_num!=0){
  	  Flit* flit = buffer_list[tag]->read();
  	  if(state[tag]==2 && flit->sched_time < cycles && !out_port->out_link->rInPort->buffer_list[out_vc[tag]]->isFull()){
  	      buffer_list[tag]->dequeue();
  	      flit->vc = out_vc[tag];
	      // added
	      flit->trace_node.push_back(NI::id);
	      flit->trace_time.push_back(cycles);
  	      out_port->buffer_list[0]->enqueue(flit);
  	      out_port->out_link->rInPort->buffer_list[out_vc[tag]]->get_credit();
  	      flit->sched_time = cycles;
  	      if(flit->type == 1 || flit->type == 10)
  		state[tag] = 0; //idle
  	      rr_priority_record = (rr_priority_record+1)%(vn_num*vc_priority_per_vn);
  	      return;
          }
      }
  }
}

void NI::dequeue(){
  if(out_port->buffer_list[0]->cur_flit_num != 0 && out_port->buffer_list[0]->read()->sched_time < cycles){
      Flit* flit = out_port->buffer_list[0]->dequeue();
	 // if(out_port->out_link->rInPort->buffer_list[flit->vc]->isFull())
	  //   cout<< "ni dequeue" << endl;
      out_port->out_link->rInPort->buffer_list[flit->vc]->enqueue(flit);
      flit->sched_time = cycles;

      if(flit->type == 0 || flit->type == 10){
		  VCRouter* vcRouter = dynamic_cast<VCRouter*>(out_port->out_link->rInPort->router_owner);
		  assert(vcRouter != NULL);
#ifdef Countlatency
		  //statistics for head flit in all types
		  if (flit->packetid * 3 + flit->packet->message.type < CountNum){
			DNN_latency[flit->packetid * 3 + flit->packet->message.type][5] = cycles;
		  }
#endif
#ifdef SHARED_VC
		  if(flit->packet->signal->QoS == 1){
			  out_port->out_link->rInPort->priority_vc.push_back(flit->vc);
			  out_port->out_link->rInPort->priority_switch.push_back(flit->vc);
		  }
#endif
		  int route_result = vcRouter->getRoute(flit);
		  out_port->out_link->rInPort->out_port[flit->vc] = route_result;
		  assert(out_port->out_link->rInPort->state[flit->vc] == 1);
		  out_port->out_link->rInPort->state[flit->vc] = 2; //wait for vc allocation
      }
  }
}


void NI::inputCheck(){
  for(int i=0; i<vn_num*(vc_per_vn + vc_priority_per_vn); i++){
      ///< if the in port buffer contains flit
      if(in_port->buffer_list[i]->cur_flit_num != 0){
		  assert(in_port->state[i] == 2);
		  Flit* flit = in_port->buffer_list[i]->readLast(); ///> reference to the last flit in buffer
		  int length = (flit->packet->length-1)/FLIT_LENGTH+1;

#ifdef Countlatency
		  if (flit->type == 1 || flit->type == 10)
		  {
			  //statistics for tail flit arrived at NI in all types
			  if(flit->packetid * 3 + flit->packet->message.type < CountNum){
			  	if(DNN_latency[flit->packetid * 3 + flit->packet->message.type][6] == 0)
			  	{DNN_latency[flit->packetid * 3 + flit->packet->message.type][6] = cycles;}
			  }
		  }
#endif
		  /// normal check
		  if(flit->sched_time < cycles && (flit->type == 1 || flit->type == 10)){  ///< tail or head tail flit
			  in_port->state[i] = 0; //idle; 2->0; no need for vc allocation
			  Packet* packet = flit->packet;
#ifndef newpooling
			  packet->message.out_cycle = cycles + LINK_TIME - 1;   // 1 cycle delay from NI_network to masterNI/slaveNI
			  NI_cycle_npooling = cycles;
#endif
#ifdef newpooling
			  if (packet->message.penable > 0)
			  {
				  if (NI_cycle_npooling < cycles + 4)
				  {
					  NI_cycle_npooling = cycles + POOLING_DELAY;
					  packet->message.out_cycle = NI_cycle_npooling + LINK_TIME - 1; // pipeline the pooling unit
				  }
				  else
				  {
					  NI_cycle_npooling = NI_cycle_npooling + POOLING_DELAY;
					  packet->message.out_cycle = NI_cycle_npooling + LINK_TIME - 1; // add pooling unit delay
				  }
				  // cout << cycles << " activate pooling at cycle " << packet->message.out_cycle << endl;
			  }
			  else
			  {
				  packet->message.out_cycle = cycles + LINK_TIME - 1;   // 1 cycle delay from NI_network to masterNI/slaveNI
				  NI_cycle_npooling = cycles;
			  }
#endif
			  if(packet->type == 0){ // request type 0
				  packet_buffer_out[0].push_back(packet);
				  count_r++;
				  //cout << "NI received " << count_r << endl;
			  }
			  else{ // response type 1
				  count_r_resp++;
				  //cout <<  "NI received response " << count_r_resp << endl;
				  packet_buffer_out[1].push_back(packet);
			  }


			  if(packet->message.QoS == 3 || packet->message.QoS == 1){
			  total_delay += cycles - packet->send_out_time;
			  total_num++;
			  }else{
			  total_delay_URS += cycles - packet->send_out_time;
			  total_num_URS++;
			  }


			  int delay = cycles - packet->send_out_time - (abs(packet->message.NI_id/X_NUM-packet->destination[0])+abs(packet->message.NI_id%X_NUM-packet->destination[1])+1)*3 - 2 - ((packet->length-1)/FLIT_LENGTH+1);
			  assert(delay>=0);



#ifdef STD_LATENCY
	      if(packet->signal->test_tag == 1){
#else
	      if(packet->message.QoS==3 || packet->message.QoS==1){
#endif

		  worst_LCS = (worst_LCS<delay)?delay:worst_LCS;
		  if(delay < DISTRIBUTION_NUM-1)
		    LCS_delay_distribution[delay]++;
		  else
		    LCS_delay_distribution[DISTRIBUTION_NUM-1]++;}
	      else{
		  worst_URS = (worst_URS<delay)?delay:worst_URS;
		  if(delay < DISTRIBUTION_NUM-1)
		    URS_delay_distribution[delay]++;
		  else
		    URS_delay_distribution[DISTRIBUTION_NUM-1]++;
	      }


	      //std::cout<<"Receive packet at cycles: "<< cycles << ", at NI id: " << id << ". Packet length:" <<packet->length <<". slave id:" << packet->slave_id << ". sequence id:" << packet->sequence_id<<std::endl;

              // ideal_time = (hop+1)*2+flit_num; hop: the routers the packet get through
	      while(in_port->buffer_list[i]->cur_flit_num != 0){
			  Flit* flit = in_port->buffer_list[i]->dequeue();
			  length = (flit->packet->length-1)/FLIT_LENGTH+1;

			  delete flit;
	      }
	  }
      }
  }
}

void NI::runOneStep(){
  packetDequeue();
  vcAllocation();
  switchArbitration();

  dequeue();
  inputCheck();
}

NI::~NI (){
  FlitBuffer* flitBuffer;
  while(buffer_list.size()!=0){
      flitBuffer = buffer_list.back();
      buffer_list.pop_back();
      delete flitBuffer;
  }
  PacketBuffer* packetBuffer;
  while(packetBuffer_list.size()!=0){
      packetBuffer = packetBuffer_list.back();
      packetBuffer_list.pop_back();
      delete packetBuffer;
  }
  delete in_port;
  delete out_port;
}

