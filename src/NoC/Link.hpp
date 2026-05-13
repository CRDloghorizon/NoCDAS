/*
 * Link.hpp
 *
 */

#ifndef VC_LINK_HPP_
#define VC_LINK_HPP_

#include "Flit.hpp"
#include "RInPort.hpp"
#include "ROutPort.hpp"

class RInPort;
class ROutPort;

/*
 * @brief link: link between 2 ports: RInPort and ROutPort
 */
class Link{
public:
  Link(RInPort* t_rInPort);
  RInPort* rInPort;
  ROutPort* rOutPort;
};



#endif /* VC_LINK_HPP_ */
