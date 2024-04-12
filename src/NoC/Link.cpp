/*
 * Link.cpp
 *
 */


#include "Link.hpp"

Link::Link(RInPort* t_rInPort){
  rInPort = t_rInPort;
  rOutPort = NULL;
}
