/*
 * Router.hpp
 */

#ifndef VCROUTER_HPP_
#define VCROUTER_HPP_

#include "FlitBuffer.hpp"
#include "Link.hpp"
#include "RInPort.hpp"
#include "ROutPort.hpp"
#include "VCNetwork.hpp"
#include "NRBase.hpp"
#include <vector>

class RInPort;
class ROutPort;
class VCNetwork;

extern unsigned int cycles;

class VCRouter: public NRBase
{
public:

  VCRouter (int* t_id, int in_out_port_num, VCNetwork* t_vcNetwork, int t_vn_num, int t_vc_per_vn, int t_vc_priority_per_vn, int t_in_depth);

  // Main methods
  int getRoute(Flit* t_flit);
  void vcRequest();
  void getSwitch();
  void outPortDequeue();

  /** @brief To run Routing, VC_allocation, Switching
   *
   *    vcRequest();
   *    getSwitch();
   *    outPortDequeue();
   */
  void runOneStep();

  // Main components
  std::vector<RInPort*> in_port_list;
  std::vector<ROutPort*> out_port_list;



  // Network
  VCNetwork* vcNetwork;
  int id[2];

  int rr_port;
  int port_num;

  int port_total_utilization;
  int port_utilization_innet;

  ~VCRouter ();
};

#endif /* VCROUTER_HPP_ */
