/*
 * ROutPort.hpp
 *
 *  Created on: 2019年8月22日
 *      Author: wr
 */

#ifndef VC_ROUTPORT_HPP_
#define VC_ROUTPORT_HPP_

#include "Link.hpp"
#include "Port.hpp"

class Link;


class ROutPort : public Port
{
public:
  ROutPort (int t_id, int t_vn_num, int t_vc_per_vn, int t_vc_priority_per_vn, int t_depth);
  ~ROutPort();
  Link * out_link;
};

#endif /* VC_ROUTPORT_HPP_ */
