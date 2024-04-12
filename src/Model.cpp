/*
 * Model.cpp
 * 
 */


#include "Model.hpp"
#include <math.h>
#include <ctime>
#include <string>
#include <sstream>
#include <iomanip>

Model::Model()
{
	layernum = 0;
}


bool Model::load()// load model
{
	cout<<"model file loading (filename: " << GlobalParams::NNmodel_filename << ")..."<< endl;
	ifstream fin(GlobalParams::NNmodel_filename, ios::in);
	char temp_type[20], temp_actfun[10];
	int temp;
	int temp_c_x, temp_c_y, temp_i, temp_z, temp_std, temp_x, temp_y, temp_pad;
	deque< deque< int > > conv;
	deque< deque< int > > pool;
	int all_Nue = 0;
	layernum = 0;
	// *****************model setting*******************
	cout<<endl;
	cout<<"layer_ID |    type | Neu_num |       X |       Y | channel |   C/P_X |   C/P_Y |  stride | padding | act_fun |"<<endl;
	cout<<"--------------------------------------------------------------------------------------------------------------"<<endl;
	while(fin >> temp_type){

			if (!strcmp( temp_type, "Input"))
			{
				all_layer_type.push_back('i');
				char line[256];
				fin.getline(line, sizeof(line) - 1);
				sscanf(line, "%d %d %d", &temp_x, &temp_y, &temp_z); // input x, input y, input channel
				temp = temp_x * temp_y * temp_z;
				deque< int > temp_layer_size;
				temp_layer_size.push_back(temp);
				temp_layer_size.push_back(temp_x);	//input x
				temp_layer_size.push_back(temp_y);	//input y
				temp_layer_size.push_back(temp_z);	//input ch

				all_layer_size.push_back(temp_layer_size);
				layernum++;
				all_Nue+=temp;
				cout<<setw(8)<<all_layer_type.size()-1<<" |"<<      setw(8)<<"Input"<<" |"<<setw(8)<<temp_layer_size[0]<<" |"
					<<setw(8)<<temp_layer_size[1]<<" |"<<setw(8)<<temp_layer_size[2]<<" |"<<setw(8)<<temp_layer_size[3]<<" |"
					<<                   setw(10)<<" |"<<                   setw(10)<<" |"<<                   setw(10)<<" |"
					<<                   setw(10)<<" |"<<                   setw(10)<<" |"<<endl;
			}
			else if (!strcmp(temp_type, "Conv2D"))
			{
				all_layer_type.push_back('c');
				char line[256];
				fin.getline(line, sizeof(line) - 1);
				sscanf(line, "%d %d %d %d %s %d %d", &temp_i, &temp_c_x, &temp_c_y, &temp_z, &temp_actfun, &temp_pad, &temp_std); // input channel, Kernel x, Kernel y, output Channel.
				temp = temp_i*temp_z;
				deque< int > temp_layer_size;
				temp_layer_size.push_back(temp);
				temp_layer_size.push_back(temp_c_x);
				temp_layer_size.push_back(temp_c_y);
				temp_layer_size.push_back(temp_z);
				temp_layer_size.push_back(temp_i);

				if(!strcmp( temp_actfun, "relu"))
					temp_layer_size.push_back(RELU); // 1
				else if(!strcmp( temp_actfun, "tanh"))
					temp_layer_size.push_back(TANH); // 2
				else if(!strcmp( temp_actfun, "sigmoid"))
					temp_layer_size.push_back(SIGMOID); // 3
				else if(!strcmp( temp_actfun, "linear"))
					temp_layer_size.push_back(LINEAR); // 0
				temp_layer_size.push_back(temp_pad);
				temp_layer_size.push_back(temp_std);

				layernum++;
				all_layer_size.push_back(temp_layer_size);
				all_Nue+=temp;
				cout<<setw(8)<<all_layer_type.size()-1<<" |"<<     setw(8)<<"Conv2D"<<" |"<<setw(8)<<temp_layer_size[0]<<" |"
					<<setw(8)<<temp_layer_size[1]<<" |"<<setw(8)<<temp_layer_size[2]<<" |"<<setw(8)<<temp_layer_size[3]<<" |"
					<<                   setw(10)<<" |"<<                   setw(10)<<" |"<<        setw(8) << temp_std<<" |"
					<<         setw(8)<<temp_pad <<" |"<<	setw(8)<<temp_actfun	<<" |"<<endl;
			}
			else if (!strcmp( temp_type, "Pool"))
			{
				all_layer_type.push_back('p');
				char line[256];
				fin.getline(line, sizeof(line) - 1);
				sscanf(line, "%d %d %d %d %d %d", &temp_i, &temp_c_x, &temp_c_y, &temp_z, &temp_pad, &temp_std); // input channel, Kernel x, Kernel y, output Channel.
				deque< int > temp_layer_size;
				temp_layer_size.push_back(temp_i);
				temp_layer_size.push_back(temp_c_x);
				temp_layer_size.push_back(temp_c_y);
				temp_layer_size.push_back(temp_z);
				temp_layer_size.push_back(temp_pad);
				temp_layer_size.push_back(temp_std);
				layernum++;
				all_layer_size.push_back(temp_layer_size);
				all_Nue+=temp;
				cout<<setw(8)<<all_layer_type.size()-1<<" |"<<       setw(8)<<"Pool"<<" |"<<setw(8)<<temp_layer_size[0]<<" |"
					<<setw(8)<<temp_layer_size[1]<<" |"<<setw(8)<<temp_layer_size[2]<<" |"<<setw(8)<<temp_layer_size[3]<<" |"
					<<                   setw(10)<<" |"<<                   setw(10)<<" |"<<        setw(8) << temp_std<<" |"
					<<        setw(8)<< temp_pad <<" |"<<                   setw(10)<<" |"<<endl;
			}
			else if (!strcmp( temp_type, "Dense"))
			{
				all_layer_type.push_back('f');
				char line[256];
				fin.getline(line, sizeof(line) - 1);
				sscanf(line, "%d %d %s", &temp_z, &temp, &temp_actfun); // input size, output size, activation
				deque< int > temp_layer_size;
				temp_layer_size.push_back(temp_z);
				temp_layer_size.push_back(temp);
				if(!strcmp( temp_actfun, "relu"))
					temp_layer_size.push_back(RELU); // 1
				else if(!strcmp( temp_actfun, "tanh"))
					temp_layer_size.push_back(TANH); // 2
				else if(!strcmp( temp_actfun, "sigmoid"))
					temp_layer_size.push_back(SIGMOID); // 3
				else if(!strcmp( temp_actfun, "linear"))
					temp_layer_size.push_back(LINEAR); // 0
				layernum++;
				all_layer_size.push_back(temp_layer_size);
				all_Nue += temp;
				cout<<setw(8)<<all_layer_type.size()-1<<" |"<<   setw(8)<<"Fully"<<" |"<<setw(8)<<temp_layer_size[1]<<" |"
					<<setw(8)<<temp_layer_size[0]<<" |"<<                setw(10)<<" |"<<                   setw(10)<<" |"
					<<                   setw(10)<<" |"<<           	 setw(10)<<" |"<<                   setw(10)<<" |"
					<<                   setw(10)<<" |"<<	 setw(8)<<temp_actfun<<" |"<<endl;
			}
			else if (!strcmp( temp_type, "%"))
			{

				char line[256];
				fin.getline(line, sizeof(line) - 1);
			}
			else
			{
				cout<<"!!Error model format: "<<temp_type<<" !!"<<endl;
				char line[256];
				fin.getline(line, sizeof(line) - 1);
			}

		}
		cout<<"Model layer loading complete"<<endl;
		cout<<"All neurons:"<<all_Nue<<endl;
		fin.close();



		return true;
}



bool Model::loadin()// load model
{
	//***************** Input setting ******************
	fstream fin_in(GlobalParams::NNinput_filename, ios::in);
	string line;
	int input_size_ch = 0;
	int j = 0;

	all_data_in.push_back(deque<float>());

	while(getline(fin_in, line))
	{
		float temp_in;
		stringstream ss(line);

		while(ss >> temp_in)
		{
			all_data_in[input_size_ch].push_back(temp_in);
		}
		j++;
		//cout << "reach2" << endl;
		if (j == all_layer_size[0][2]) {input_size_ch = input_size_ch + 1; j = 0; all_data_in.push_back(deque<float>());}
	}
	cout << all_data_in[0].size() << " with input size " << all_layer_size[0][1] << " " <<  all_layer_size[0][2] << endl;
	assert((input_size_ch == all_layer_size[0][3]) && "Input channel file is not correct!");
	//cout << "input channel read " << input_size_ch  << endl;
	fin_in.close();
	return true;
}

// random generated floats -0.5 to 0.5
float Model::generateRandomFloat() {
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f;
}

bool Model::randomin()// load model with random input
{
	//***************** Input setting ******************
	srand(unsigned(time(0)));  //NULL srand(static_cast<unsigned>(time(0)));

	int input_size_ch = 0;
	int j = 0; // y
	int input_re_ch = all_layer_size[0][2] * all_layer_size[0][3];

	all_data_in.push_back(deque<float>());

	for (int m = 0; m < input_re_ch; m++ )
	{
		for (int n = 0; n < all_layer_size[0][1]; n++ )
		{
			all_data_in[input_size_ch].push_back(generateRandomFloat()); // get x number
		}
		j++; // y + 1
		if (j == all_layer_size[0][2]) {input_size_ch = input_size_ch + 1; j = 0; all_data_in.push_back(deque<float>());}
	}
	//cout << input_size_ch << ' ' << all_layer_size[0][3] << endl;
	cout << all_data_in[0].size() << " with random input, size " << all_layer_size[0][1] << " " <<  all_layer_size[0][2] << endl;
	assert((input_size_ch == all_layer_size[0][3]) && "Input channel of RandomIn is not correct!");
	//cout << "input channel read " << input_size_ch  << endl;

	return true;
}

	//***************** Weight setting ******************

bool Model::loadweight()// load model
{
	//***************** Input setting ******************
	fstream fin_w(GlobalParams::NNweight_filename, ios::in);

	string line;
	int j = 0;
	while(getline(fin_w, line))
	{
		float temp_in;
		stringstream ss(line);
		all_weight_in.push_back(deque<float>());
		while(ss >> temp_in)
		{
			all_weight_in[j].push_back(temp_in);
		}
//		cout << all_weight_in[j].size() << endl;
		j++;
	}
	cout << "Load weight " << j << " (lines) with weight size " << all_weight_in.size() << endl;

	fin_w.close();
	return true;
}

bool Model::randomweight()// load model
{
	//***************** Input setting ******************

	int num_w, len_w;
	int clayer = all_layer_type.size();
	int j = 0;
	int check = 0;

	for(int i = 0; i < clayer; i++)
	{
		
		if(all_layer_type[i] == 'c')
		{
			num_w = all_layer_size[i][3]; //z output ch
			len_w = all_layer_size[i][4] * all_layer_size[i][1] * all_layer_size[i][2] + 1;
			check = check + num_w;

			for (int m = 0; m < num_w; m++)
			{
				all_weight_in.push_back(deque<float>());
				for (int n = 0; n < len_w; n++)
				{
					all_weight_in[j].push_back(generateRandomFloat()); // random generated
				}
				j++; // y + 1
			}
		}
		else if(all_layer_type[i] == 'f')
		{
			num_w = all_layer_size[i][1]; //output size
			len_w = all_layer_size[i][0] + 1; //input size
			check = check + num_w;
			for (int m = 0; m < num_w; m++)
			{
				all_weight_in.push_back(deque<float>());
				for (int n = 0; n < len_w; n++)
				{
					all_weight_in[j].push_back(generateRandomFloat()); // random generated
				}
				j++; // y + 1
			}
		}
	}
	assert((check == all_weight_in.size()) && "Input channel of RandomWeight is not correct!");
	cout << "Load random weight " << j << " (lines) with weight size " << check << endl;
	return true;
}

Model::~Model (){


}
