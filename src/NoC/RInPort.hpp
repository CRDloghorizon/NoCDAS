/*
 * Port.hpp
 *
 */

#ifndef VC_RINPORT_HPP_
#define VC_RINPORT_HPP_

#include "Flit.hpp"
#include "FlitBuffer.hpp"
#include "Link.hpp"
#include "Port.hpp"
#include "VCRouter.hpp"
#include "NRBase.hpp"
#include "../parameters.hpp"

#include <cassert>
#include <vector>


class FlitBuffer;
class VCRouter;
class Link;

extern unsigned int cycles;

class RInPort : public Port{
public:

  RInPort(int t_id, int t_vn_num, int t_vc_per_vn, int t_vc_priority_per_vn, int t_depth, NRBase * t_owner, Link * t_in_link);

  void vc_request();

  int vc_allocate(Flit*);
  int vc_allocate_normal(Flit*);

  int vc_allocate_priority(int);

  void getSwitch();

  ~RInPort();

  //GROPC field
  std::vector<int> state; //0->I; 1->R; 2-> V; 3->A;
  std::vector<int> out_port;
  std::vector<int> out_vc;

  //Switch arbitration
  int rr_record; //round robin record


  // for shared priority
  std::vector<int> priority_vc;
  int count_vc;
  std::vector<int> priority_switch;
  int count_switch;


  // for individual priority
  int rr_priority_record;
  int starvation;

  NRBase * router_owner;
  Link * in_link;

  // added
  int rid[2];
  //added oct20
  int flitOperNuminOneCycle;
};



#endif /* VC_RINPORT_HPP_ */
