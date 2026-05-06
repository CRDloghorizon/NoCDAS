/*
 * VCRouter.hpp
 */

#ifndef VCROUTER_HPP_
#define VCROUTER_HPP_

#include "FlitBuffer.hpp"
#include "Link.hpp"
#include "RInPort.hpp"
#include "ROutPort.hpp"
#include "VCNetwork.hpp"
#include "NRBase.hpp"
#include "Model.hpp"
#include <vector>
#include <map>

class RInPort;
class ROutPort;
class VCNetwork;

extern unsigned int cycles;

class VCRouter: public NRBase
{
public:
  // Struct that models the hardware state registers of a VC for cNoC
  struct ComputeVCState {
      int compute_op;
      double running_max;
      double running_sum;
      bool is_active;

      ComputeVCState() : compute_op(-1), running_max(-1e9), running_sum(0.0), is_active(false) {}
      
      void reset() {
          compute_op = -1;
          running_max = -1e9;
          running_sum = 0.0;
          is_active = false;
      }
  };

  VCRouter (int* t_id, int in_out_port_num, VCNetwork* t_vcNetwork, int t_vn_num, int t_vc_per_vn, int t_vc_priority_per_vn, int t_in_depth);

  // Main methods
  int getRoute(Flit* t_flit);
  void vcRequest();
  void getSwitch();
  void outPortDequeue();

  /** @brief To run Routing, VC_allocation, Switching
   *
   * vcRequest();
   * getSwitch();
   * outPortDequeue();
   */
  void runOneStep();

  // Local SRAM (replaces simple W registers to support Transformer operations)
  std::vector<float> local_weights;           // Weights distributed to this router (e.g., Q, K, V projections)
  std::vector<float> local_kv_cache;

  // Indexes: [port_idx][vc_idx]
  std::vector<std::vector<ComputeVCState>> vc_compute_state;

  int kv_token_count = 0;

  int current_sram_usage;
  
  bool allocateSRAM(int num_floats);
  void clearSRAM();
  void storeKV(float kv_value);
  void writeKV(int index, float kv_value);
  void storeWeight(float weight_value);

//   std::vector<unsigned int> mfu_occupied_until;
  unsigned int mfu_occupied_until;
  
  // List of indexes of output tasks assigned to this router
  std::vector<int> assigned_tasks;
  
  // Multi-way Function Unit (MFU) functions
  void computeInTransit(Flit* t_flit, int port_idx);
  void processDistributionPacket(Flit* t_flit);

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

  int rr_out_port;

  ~VCRouter ();
};

#endif /* VCROUTER_HPP_ */