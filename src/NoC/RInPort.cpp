/*
 * Port.cpp
 *
 */

#include "RInPort.hpp"


RInPort::RInPort(int t_id, int t_vn_num, int t_vc_per_vn, int t_vc_priority_per_vn, int t_depth, NRBase * t_owner, Link * t_in_link)
   :Port(t_id, t_vn_num, t_vc_per_vn, t_vc_priority_per_vn, t_depth)
{
  for(int i=0; i<vn_num*(vc_per_vn+vc_priority_per_vn); i++){
      state.push_back(0);
      out_port.push_back(-1);
      out_vc.push_back(-1);
  }
  rr_record = 0;
  rr_priority_record = 0;
  router_owner = t_owner;
  in_link = t_in_link;
  count_vc = 0;
  count_switch = 0;
  starvation = 0;
  //added oct20
  flitOperNuminOneCycle = 0;
}


int RInPort::vc_allocate(Flit* t_flit){
  int vn = t_flit->vnet;

  for(int i=0; i<vc_per_vn; i++){
      if(state[vn*vc_per_vn+i] == 0){ //idle
	  state[vn*vc_per_vn+i] = 1;  //wait for routing
	  return vn*vc_per_vn+i;  //vc id
      }
  }
  return -1; // no vc available
}

int RInPort::vc_allocate_normal(Flit* t_flit){
  int vn = t_flit->vnet;

  for(int i=0; i<vc_per_vn-vc_priority_per_vn; i++){
      if(state[vn*vc_per_vn+i] == 0){ //idle
	  state[vn*vc_per_vn+i] = 1;  //wait for routing
	  return vn*vc_per_vn+i;  //vc id
      }
  }
  return -1; // no vc available
}

int RInPort::vc_allocate_priority(int vn_rank){

  for(int i=0; i<vc_priority_per_vn; i++){
      int tag = vn_num*vc_per_vn+vc_priority_per_vn*vn_rank+i;
      if(state[tag] == 0){
          state[tag] = 1;
          return tag;
      }
  }
  return -1;
}




void RInPort::vc_request(){
	flitOperNuminOneCycle = 0;
  // for priority packet (shared VCs) QoS = 1
#ifdef SHARED_VC
   std::vector<int>::iterator iter;
   for(iter=priority_vc.begin(); iter<priority_vc.end();){
       if(count_vc == STARVATION_LIMIT)
 	  break;
       int tag = (*iter);
       if(state[tag] == 2){
 	  Flit* flit = buffer_list[tag]->flit_queue.front();
 	  assert(flit->type == 0 || flit->type == 10);
 	  VCRouter* vcRouter = dynamic_cast<VCRouter*>(router_owner);
 	  assert(vcRouter != NULL);
 	  int vc_result = vcRouter->out_port_list[out_port[tag]]->out_link->rInPort->vc_allocate(flit);
 	  if (vc_result != -1){
 	      //vcRouter->out_port_list[out_port[tag]]->out_link->rInPort->priority_vc.push_back(vc_result);
 	      //vcRouter->out_port_list[out_port[tag]]->out_link->rInPort->priority_switch.push_back(vc_result);
 	      state[tag] = 3; //active
 	      out_vc[tag] = vc_result; //record the output vc in the streaming down node
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
       if(state[tag] == 2){
       	  Flit* flit = buffer_list[tag]->flit_queue.front();
       	  assert(flit->type == 0 || flit->type == 10);
       	  VCRouter* vcRouter = dynamic_cast<VCRouter*>(router_owner);
       	  assert(vcRouter != NULL);
       	  int vn_rank = (tag-vn_num*vc_per_vn)/vc_priority_per_vn;
       	  int vc_result = vcRouter->out_port_list[out_port[tag]]->out_link->rInPort->vc_allocate_priority(vn_rank);
       	  if (vc_result != -1){
       	      state[tag] = 3; //active
       	      out_vc[tag] = vc_result; //record the output vc in the streaming down node
       	  }
       }
   }

   // for URS packet
  for(int i=0; i<vn_num*vc_per_vn; i++){
      int tag = (i+rr_record)%(vn_num*vc_per_vn);
      if(state[tag] == 2){
	  Flit* flit = buffer_list[tag]->flit_queue.front();
	  assert(flit->type == 0 || flit->type == 10);
	  VCRouter* vcRouter = dynamic_cast<VCRouter*>(router_owner);
	  assert(vcRouter != NULL);
#ifdef SHARED_PRI
	  int vc_result = vcRouter->out_port_list[out_port[tag]]->out_link->rInPort->vc_allocate_normal(flit);
#else
	  int vc_result = vcRouter->out_port_list[out_port[tag]]->out_link->rInPort->vc_allocate(flit);
#endif
	  if (vc_result != -1){
	      state[tag] = 3; //active
	      out_vc[tag] = vc_result; //record the output vc in the streaming down node
	  }
      }
  }

  // in case of no normal packet
#ifdef SHARED_VC

   for(; iter<priority_vc.end();){
       int tag = (*iter);
       if(state[tag] == 2){
 	  Flit* flit = buffer_list[tag]->flit_queue.front();
 	  assert(flit->type == 0 || flit->type == 10);
 	  VCRouter* vcRouter = dynamic_cast<VCRouter*>(router_owner);
 	  assert(vcRouter != NULL); // only vc router will call this methed
 	  int vc_result = vcRouter->out_port_list[out_port[tag]]->out_link->rInPort->vc_allocate(flit);
 	  if (vc_result != -1){
 	      //vcRouter->out_port_list[out_port[tag]]->out_link->rInPort->priority_vc.push_back(vc_result);
 	      //vcRouter->out_port_list[out_port[tag]]->out_link->rInPort->priority_switch.push_back(vc_result);
 	      state[tag] = 3; //active
 	      out_vc[tag] = vc_result; //record the output vc in the streaming down node
 	      iter = priority_vc.erase(iter);
 	  }else
    	    iter++;
       }else
	 iter++;
   }
#endif

}



void RInPort::getSwitch(){

  // for priority packet
#ifdef SHARED_VC
  std::vector<int>::iterator iter;
  for(iter=priority_switch.begin(); iter<priority_switch.end();iter++){
      if(count_switch == STARVATION_LIMIT)
	  break;
      int tag = (*iter);
      if(buffer_list[tag]->cur_flit_num > 0 && state[tag] == 3 && buffer_list[tag]->read()->sched_time < cycles){
	  Flit* flit = buffer_list[tag]->read();
	  flit->vc = out_vc[tag];
	  VCRouter* vcRouter = dynamic_cast<VCRouter*>(router_owner);
	  assert(vcRouter != NULL); // only VC router will call this method
	  if(!vcRouter->out_port_list[out_port[tag]]->out_link->rInPort->buffer_list[flit->vc]->isFull()){
	      buffer_list[tag]->dequeue();

	      //if(vcRouter->out_port_list[out_port[tag]]->buffer_list[0]->isFull())
	     	    // cout<< "router switch" << endl;
	      // added
	      flit->trace_node.push_back(vcRouter->id[0]*X_NUM + vcRouter->id[1]);
	      flit->trace_time.push_back(cycles);
	      vcRouter->out_port_list[out_port[tag]]->buffer_list[0]->enqueue(flit);
	      vcRouter->out_port_list[out_port[tag]]->out_link->rInPort->buffer_list[flit->vc]->get_credit();
	      flit->sched_time = cycles;
	      count_switch++;
	      if(flit->type == 1 || flit->type == 10 ){
		  state[tag] = 0; //idle
		  priority_switch.erase(iter);
	      }
	      return;
	  }
      }
  }
  count_switch = 0;
#endif

  // for LCS packet (individual VC)
  for(int i=vn_num*vc_per_vn; i<vn_num*(vc_per_vn+vc_priority_per_vn); i++){ //vc round robin; pick up non-empty buffer with state A (3)
      if(starvation == STARVATION_LIMIT){
    	  starvation = 0;
      	  break;
      }
      int tag = (i+rr_priority_record)%(vn_num*vc_priority_per_vn)+vn_num*vc_per_vn;
      if(buffer_list[tag]->cur_flit_num > 0 && state[tag] == 3 && buffer_list[tag]->read()->sched_time < cycles){
	  Flit* flit = buffer_list[tag]->read();
	  flit->vc = out_vc[tag];
	  VCRouter* vcRouter = dynamic_cast<VCRouter*>(router_owner);
	  assert(vcRouter != NULL); // only vc router will call this methed
	  //added oct20
	  if(!vcRouter->out_port_list[out_port[tag]]->out_link->rInPort->buffer_list[flit->vc]->isFull() && flitOperNuminOneCycle == 0){
	      buffer_list[tag]->dequeue();
	      // added
//	      flit->trace_node.push_back(vcRouter->id[0]*X_NUM + vcRouter->id[1]);
//	      flit->trace_time.push_back(cycles);
	      vcRouter->out_port_list[out_port[tag]]->buffer_list[0]->enqueue(flit);
	      // added oct20
	      flitOperNuminOneCycle = flitOperNuminOneCycle + 1; //do one enqueue
	      vcRouter->out_port_list[out_port[tag]]->out_link->rInPort->buffer_list[flit->vc]->get_credit();
	      flit->sched_time = cycles;
	      if(flit->type == 1 || flit->type == 10 ){
	    	  state[tag] = 0; //idle
	      }
	      rr_priority_record = (rr_priority_record+1) % (vn_num*vc_priority_per_vn);
	      starvation++;
	      return;
	  }
      }
  }



  // for normal packet
  for(int i=0; i<vn_num*vc_per_vn; i++){ //vc round robin; pick up non-empty buffer with state A (3)
      int tag = (i + rr_record) % (vn_num * vc_per_vn);
      if(buffer_list[tag]->cur_flit_num > 0 && state[tag] == 3 && buffer_list[tag]->read()->sched_time < cycles){
		  Flit* flit = buffer_list[tag]->read();
		  flit->vc = out_vc[tag];
		  VCRouter* vcRouter = dynamic_cast<VCRouter*>(router_owner);
		  assert(vcRouter != NULL); // only vc router will call this methed
#ifdef outPortNoInfinite
		  if (!vcRouter->out_port_list[out_port[tag]]->out_link->rInPort->buffer_list[flit->vc]->isFull()
				&& !vcRouter->out_port_list[out_port[tag]]->buffer_list[0]->isFull()
				&& flitOperNuminOneCycle == 0)
#else
		  if(!vcRouter->out_port_list[out_port[tag]]->out_link->rInPort->buffer_list[flit->vc]->isFull())
#endif
		  {
			  buffer_list[tag]->dequeue();
			  // added
	//	      flit->trace_node.push_back(vcRouter->id[0]*X_NUM + vcRouter->id[1]);
	//	      flit->trace_time.push_back(cycles);
			  // check wrong switch allocation
			  vcRouter->out_port_list[out_port[tag]]->buffer_list[0]->enqueue(flit);
			  //added oct20
			  flitOperNuminOneCycle = flitOperNuminOneCycle + 1; //do one enqueue
			  vcRouter->out_port_list[out_port[tag]]->buffer_list[0]->get_credit();
			  //
			  vcRouter->out_port_list[out_port[tag]]->out_link->rInPort->buffer_list[flit->vc]->get_credit();
			  flit->sched_time = cycles;
			  if(flit->type == 1 || flit->type == 10){
				  state[tag] = 0; //idle
			  }
			  rr_record = (rr_record+1)%(vn_num*vc_per_vn);
			  return;
		  }
      }
  }

  // in case of no normal packet
#ifdef SHARED_VC

  for(; iter<priority_switch.end();iter++){
      int tag = (*iter);
      if(buffer_list[tag]->cur_flit_num > 0 && state[tag] == 3 && buffer_list[tag]->read()->sched_time < cycles){
	  Flit* flit = buffer_list[tag]->read();
	  flit->vc = out_vc[tag];
	  VCRouter* vcRouter = dynamic_cast<VCRouter*>(router_owner);
	  assert(vcRouter != NULL); // only vc router will call this methed
	  if(!vcRouter->out_port_list[out_port[tag]]->out_link->rInPort->buffer_list[flit->vc]->isFull()){
	      buffer_list[tag]->dequeue();

	      //if(vcRouter->out_port_list[out_port[tag]]->buffer_list[0]->isFull())
	     	    // cout<< "router switch" << endl;
	      // added
	      flit->trace_node.push_back(vcRouter->id[0]*X_NUM + vcRouter->id[1]);
	      flit->trace_time.push_back(cycles);
	      vcRouter->out_port_list[out_port[tag]]->buffer_list[0]->enqueue(flit);
	      vcRouter->out_port_list[out_port[tag]]->out_link->rInPort->buffer_list[flit->vc]->get_credit();
	      flit->sched_time = cycles;
	      if(flit->type == 1 || flit->type == 10 ){
			  state[tag] = 0; //idle
			  priority_switch.erase(iter);
	      }
	      return;
	  }
      }
  }
#endif

  // for LCS packet (individual VC)
  for(int i=vn_num*vc_per_vn;i<vn_num*(vc_per_vn+vc_priority_per_vn); i++){ //vc round robin; pick up non-empty buffer with state A (3)
      int tag = (i+rr_priority_record)%(vn_num*vc_priority_per_vn)+vn_num*vc_per_vn;
      if(buffer_list[tag]->cur_flit_num > 0 && state[tag] == 3 && buffer_list[tag]->read()->sched_time < cycles){
	  Flit* flit = buffer_list[tag]->read();
	  flit->vc = out_vc[tag];
	  VCRouter* vcRouter = dynamic_cast<VCRouter*>(router_owner);
	  assert(vcRouter != NULL);
	  if(!vcRouter->out_port_list[out_port[tag]]->out_link->rInPort->buffer_list[flit->vc]->isFull()){
	      buffer_list[tag]->dequeue();
	      // added
	      flit->trace_node.push_back(vcRouter->id[0]*X_NUM + vcRouter->id[1]);
	      flit->trace_time.push_back(cycles);
	      vcRouter->out_port_list[out_port[tag]]->buffer_list[0]->enqueue(flit);
	      vcRouter->out_port_list[out_port[tag]]->out_link->rInPort->buffer_list[flit->vc]->get_credit();
	      flit->sched_time = cycles;
	      if(flit->type == 1 || flit->type == 10 ){
	    	  state[tag] = 0; //idle
	      }
	      rr_priority_record = (rr_priority_record+1)%(vn_num*vc_priority_per_vn);
	      return;
	  }
      }
  }
}

RInPort::~RInPort(){

}




