/*
 * Model.hpp
 * DNN model input
 */

#ifndef MODEL_HPP_
#define MODEL_HPP_

#include <iostream>
#include <fstream>
#include <vector>
#include <deque>
#include <iomanip>
#include <algorithm>    //std::random_shuffle
#include <ctime>        //std::time
#include <cstdlib>		//std::rand, std::srand
#include <stdio.h>
#include <string.h>
#include <cassert>
#include "parameters.hpp"

using namespace std;

#define LINEAR              0
#define RELU             	1
#define TANH              	2
#define SIGMOID             3
#define SOFTMAX             4

struct NeuInfo
{
	int ID_Neu;			// ID of the Neuron in software
    int ID_layer;			// Layer Number of the Neuron
	char Type_layer;			// Type of the layer
	deque< float> weight;		// Weight of the Neuron
	int ID_In_layer;		// ID of the Neuron in the layer
	int local_x;
	int local_y;
	int local_ch;
	int sta_x;
	int end_x;
	int sta_y;
	int end_y;
	int ID_conv;
};

class Model
{
public:
	int layernum;

    Model();

	float generateRandomFloat();

    bool load();

    bool loadin();

    bool randomin();

    bool loadweight();

	bool randomweight();

	deque< char > all_layer_type;
	deque< deque< int > > all_layer_size;
	deque< deque< float > > all_data_in;
	deque< deque< float > > all_weight_in;
	deque< deque< float > > all_conv_weight;


	~Model();
};



#endif /* MODEL_HPP_ */
