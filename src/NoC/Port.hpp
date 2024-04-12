/*
 * Port.hpp
 *
 */

#ifndef VC_PORT_HPP_
#define VC_PORT_HPP_

#include "Flit.hpp"
#include "FlitBuffer.hpp"
#include <vector>


class Port
{
public:
  Port(int t_id, int t_vn_num, int t_vc_per_vn, int t_vc_priority_per_vn, int t_depth);
  virtual ~Port();

  int id;
  int vn_num;
  int vc_per_vn;
  int depth;
  int vc_priority_per_vn;

  // added
  int count_flit;

  std::vector<FlitBuffer*> buffer_list;
};

#endif /* VC_PORT_HPP_ */
