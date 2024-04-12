/*
 * Network.cpp
 */

#include "VCNetwork.hpp"


VCNetwork::VCNetwork (int router_num, int router_num_x, int NI_num_total, int* NI_num, int t_vn_num, int t_vc_per_vn, int t_vc_priority_per_vn, int t_in_depth)
{
  cout << "Object is being created" << endl;
  int NI_seq = 0;
  routerNum = router_num;
  NINum = NI_num_total;
  router_list.reserve(router_num);
  NI_list.reserve(NI_num_total);

  for(int i=0; i<router_num; i++){ // routers = number of MAC nodes
      int id[2];
      id[0] = i/router_num_x;
      id[1] = i%router_num_x;
      int in_out_port_num = 4 + NI_num[i]; //NI_num[i] = 1, in total 64 NI
      VCRouter* vcRouter = new VCRouter(id, in_out_port_num, this, t_vn_num, t_vc_per_vn, t_vc_priority_per_vn, t_in_depth);
      router_list.push_back(vcRouter);

      // firstly connect NI to corresponding router
      for(int j=0; j<NI_num[i]; j++){
	  NI* ni = new NI(NI_seq, this, t_vn_num, t_vc_per_vn, t_vc_priority_per_vn, t_in_depth, NI_num);
	  NI_list.push_back(ni);
	  NI_seq++; //NI id
	  // connect NI's outport with router's inport
	  ni->out_port->out_link = vcRouter->in_port_list[4+j]->in_link;
	  ni->out_port->out_link->rOutPort = ni->out_port;

	  // connect NI's inport with router's outport
	  vcRouter->out_port_list[4+j]->out_link = ni->in_port->in_link;
	  vcRouter->out_port_list[4+j]->out_link->rOutPort = vcRouter->out_port_list[4+j];
      }
  }
  // secondly, connect routers with each other according to the network size.
  for(int i=0; i<router_num; i++){
      int x,y; // (x,y)
      x = i/router_num_x;
      y = i%router_num_x;

      // port: 0->up; 1->right; 2->down; 3->left. each router connects to its right and down routers
      // port 1, right;
      if(y+1 < router_num_x){ // The right router exits;
	  VCRouter* self = router_list[i];
	  VCRouter* next = router_list[i+1];
	  // connect self's outport with next's inport
	  self->out_port_list[1]->out_link = next->in_port_list[3]->in_link;
	  self->out_port_list[1]->out_link->rOutPort = self->out_port_list[1];

	  // connect next's outport with self's inport
	  next->out_port_list[3]->out_link = self->in_port_list[1]->in_link;
	  next->out_port_list[3]->out_link->rOutPort = next->out_port_list[3];
      }
      // port 2, down;
      if(x+1 < router_num/router_num_x){ // The down router exits;
	  VCRouter* self = router_list[i];
	  VCRouter* next = router_list[i+router_num_x];
	  // connect self's outport with next's inport
	  self->out_port_list[2]->out_link = next->in_port_list[0]->in_link;
	  self->out_port_list[2]->out_link->rOutPort = self->out_port_list[2];

	  // connect next's outport with self's inport
	  next->out_port_list[0]->out_link = self->in_port_list[2]->in_link;
	  next->out_port_list[0]->out_link->rOutPort = next->out_port_list[0];
      }
  }
}

void VCNetwork::runOneStep(){
  for(int i=0; i<NINum; i++){
     // cout << "#######" << i << endl;
      NI_list[i]->runOneStep();
  }

  for(int i=0; i<routerNum; i++){
    //cout << "#######" << i << endl;
    router_list[i]->runOneStep();
  }
  // added buffer status
//  if(((int)cycles) % 1000 == 0) {show_VCR_buffer_state();}
}

// updated
int VCNetwork::port_num_f(int router){
  if(router==0 || router==(X_NUM-1) || router==(TOT_NUM-X_NUM) || router==(TOT_NUM-1))
    return 3;
  else if(router<=(X_NUM-1) || router>=(TOT_NUM-X_NUM) || (router%X_NUM)==0 || (router%X_NUM)==(X_NUM-1))
    return 4;
  return 5;
}

void VCNetwork::port_utilization(int simulate_cycles){
  for(int i=0; i<routerNum; i++){
      cout.setf(ios::fixed);
      //cout << i << " ";
      cout << setprecision(3) << ((float)router_list[i]->port_total_utilization)/(port_num_f(i)*simulate_cycles/100)<< endl;
  }
}

void VCNetwork::port_utilization_innet(int simulate_cycles){
  for(int i=0; i<routerNum; i++){
      cout.setf(ios::fixed);
      //cout << i << " ";
      cout << setprecision(3) << ((float)router_list[i]->port_utilization_innet)/((port_num_f(i)-1)*simulate_cycles/100)<< endl;
  }
}

void VCNetwork::average_LCS_latency(){
  cout.setf(ios::fixed);
  float a = 0.0;
  int b = 0;
  int len = 0;
  for(int i=0; i<NINum; i++){
      a += ((float)NI_list[i]->total_delay);
      b += NI_list[i]->total_num;
      len += NI_list[i]->num_flit;
  }
  cout << setprecision(1) << a/b << " received LCS: " << b << endl;
  cout << "Total number of flits: " << len << endl;
}

void VCNetwork::average_URS_latency(){
//  for(int i=0; i<NINum; i++){
//      cout.setf(ios::fixed);
      //cout << i << " ";
//      cout << setprecision(1) << ((float)NI_list[i]->total_delay_URS)/NI_list[i]->total_num_URS << endl;
//  }

  cout.setf(ios::fixed);
  float a = 0.0;
  int b = 0;
  for(int i=0; i<NINum; i++){
      a += ((float)NI_list[i]->total_delay_URS);
      b += NI_list[i]->total_num_URS;
  }
  cout << setprecision(1) << a/b <<  " received URS: " << b << endl;
}

void VCNetwork::show_LCS_distribution(){
  for(int i=0;i<DISTRIBUTION_NUM;i++)
      cout << NI_list[0]->LCS_delay_distribution[i] << endl;
  cout << "worst case: " << NI_list[0]->worst_LCS << endl;
}
void VCNetwork::show_URS_distribution(){
  for(int i=0;i<DISTRIBUTION_NUM;i++)
      cout << NI_list[0]->URS_delay_distribution[i] << endl;
  cout << "worst case: " << NI_list[0]->worst_URS << endl;
}

//void VCNetwork::show_VCR_buffer_state(){
//  for(int i=0; i<routerNum; i++){
//      VCRouter* temp = router_list[i];
//      router_monitor << cycles << " RID " << i;
//      for (int j=0; j<temp->port_num; j++){
//	  RInPort * tempPort = temp->in_port_list[j];
//	  router_monitor << " PID " << j << " states";
//	  for (int k: tempPort->state)
//	    {router_monitor << ' ' << k;}
//      }
//      router_monitor << endl;
//    }
//  // PID 4 NI connector
//  // PID 0-3 is 0->up; 1->right; 2->down; 3->left
//}


VCNetwork::~VCNetwork ()
{
  cout << "Object VCNetwork is being deleted" << endl;
  VCRouter* router;
  while(router_list.size()!=0){
      router = router_list.back();
      router_list.pop_back();
      delete router;
  }

  NI* ni;
  while(NI_list.size()!=0){
      ni = NI_list.back();
      NI_list.pop_back();
      delete ni;
  }
}

