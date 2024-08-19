//======================================================================================
// Name        : NoCDAS
// Author      : loghorizon
// Version     : 2024 Jul
// Copyright   : KTH
// Description : A Cycle-Accurate NoC-based Deep Neural Network Accelerator Simulator
//======================================================================================

#include <iostream>
#include <fstream>
#include <vector>
#include <deque>
#include <iomanip>
#include <stdio.h>
#include <string.h>
#include <chrono>
#include "parameters.hpp"
#include "NoC/VCNetwork.hpp"
#include "MACnet.hpp"
#include "Model.hpp"

using namespace std;

// NoC
class VCNetwork;

int packet_id;

// Statistics
vector<vector<int>> DNN_latency;


// DNN
unsigned int cycles;
int ch;
int layer;

int PE_NUM = PE_X_NUM * PE_Y_NUM;

char GlobalParams::NNmodel_filename[128] = DEFAULT_NNMODEL_FILENAME;
char GlobalParams::NNweight_filename[128] = DEFAULT_NNWEIGHT_FILENAME;
char GlobalParams::NNinput_filename[128] = DEFAULT_NNINPUT_FILENAME;

void parseCmdLine(int arg_num, char *arg_vet[])
{
    if (arg_num == 1)
	cout << "Running with default parameters" << endl;
    else {
		for (int i = 1; i < arg_num; i++) {
			if (!strcmp(arg_vet[i], "-NNmodel"))
				strcpy(GlobalParams::NNmodel_filename, arg_vet[++i]);
			else if (!strcmp(arg_vet[i], "-NNweight"))
				strcpy(GlobalParams::NNweight_filename, arg_vet[++i]);
			else if (!strcmp(arg_vet[i], "-NNinput"))
				strcpy(GlobalParams::NNinput_filename, arg_vet[++i]);
			else {
				cerr << "Error: Invalid option: " << arg_vet[i] << endl;
				exit(1);
			}
		}

    }
}

int main(int arg_num, char *arg_vet[]) {

	cout << "Initialize" << endl;
	parseCmdLine(arg_num, arg_vet);

	chrono::steady_clock::time_point begin = chrono::steady_clock::now();

	Model* cnnmodel = new Model();
	cnnmodel->load();

#ifdef fulleval
	cnnmodel->loadin();
	cnnmodel->loadweight();
#elif defined randomeval
	cnnmodel->randomin();
	cnnmodel->randomweight();
#endif	

	// statistics
	// refer to output neuron id (tmpch * ox * oy + tmpm)
#ifdef Countlatency
	DNN_latency.resize(CountNum);
	for(int i=0;i<CountNum;i++){
		DNN_latency[i].assign(8, 0);
	}
#endif
	// create vc
	packet_id = 0;
	int vn = VN_NUM;
	int vc_per_vn = VC_PER_VN;
	int vc_priority_per_vn = VC_PRIORITY_PER_VN;
	int flit_per_vc = INPORT_FLIT_BUFFER_SIZE;
	int router_num = TOT_NUM;
	int router_x_num = X_NUM;
	int NI_total = TOT_NUM; //64
	int NI_num[TOT_NUM];
	for (int i=0;i<TOT_NUM;i++){
		NI_num[i] = 1;
	}

	VCNetwork* vcNetwork = new VCNetwork(router_num, router_x_num, NI_total, NI_num, vn, vc_per_vn, vc_priority_per_vn, flit_per_vc);

	
	// create the macnet controller
	MACnet* macnet = new MACnet(PE_NUM, PE_X_NUM, PE_Y_NUM, cnnmodel, vcNetwork);

	cycles = 0;
	unsigned int simulate_cycles = 500000000;

	// Main simulation
	for(; cycles < simulate_cycles; cycles++){
		macnet->checkStatus();
		//cout << "check status: " << cycles << endl;

		if(macnet->c_layer == macnet->n_layer) break;

		macnet->runOneStep();

		vcNetwork->runOneStep();

		if(cycles%PRINT==0) {cout << "Running cycles: " << cycles << endl;}
	}

#ifdef fulleval
	for (float j: macnet->output_table[0])
	{
		cout << j << ' ';
	}
	cout << endl;
#endif

	cout << "Cycles: " << cycles << endl;

	cout << "Packet id: " << packet_id << endl;

#ifdef Countlatency
	int maxoutnum = CountNum;
	ofstream outfile_delay("./src/output/lenettrace.txt", ios::out);
	if (packet_id*3 <= CountNum) {maxoutnum = packet_id*3;}
	for(int i=0; i<maxoutnum; i++){
	  for(int j=0; j<8; j++){
		  outfile_delay << DNN_latency[i][j] << ' ';
	  }
	  outfile_delay << endl;
	}
	outfile_delay.close();
#endif
	chrono::steady_clock::time_point end = chrono::steady_clock::now();
	cout << "Execution time (sec) = " <<  (chrono::duration_cast<chrono::microseconds>(end - begin).count()) /1000000.0  << endl;

	cout << "!!END!!" << endl;
	delete macnet;
	delete cnnmodel;
	return 0;
}
