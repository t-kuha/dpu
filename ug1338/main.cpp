#include <iostream>
#include <iomanip>
#include <dirent.h>
#include <errno.h>
#include <string.h>
#include <vector>

// DNNDK
#include "n2cube.h"
#include "dputils.h"

// OpenCV
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"

// CIFAR-10
//  0: airplane										
//  1: automobile										
//  2: bird										
//  3: cat										
//  4: deer										
//  5: dog										
//  6: frog										
//  7: horse										
//  8: ship										
//  9: truck

int selects(const struct dirent *dir) 
{
	if( strstr(dir->d_name, ".png")) {
		// Filter files containing "".png"
		return 1;
	}

	return 0;
}

int main()
{
	std::cout << "------ DPU (CIFAR-10) ------" << std::endl;

	int ret = 0;

	cv::Mat img;	// Input image
	
    // See output of dnnc for input/output node name
    const char* input_node_name = "conv2d_Conv2D";
	const char* output_node_name = "dense_1_MatMul";

	// DPU
	ret = dpuOpen();
	if(ret){
		std::cerr << "[FAIL] dpuOpen()" << std::endl;
	}

	// Kernel
	DPUKernel *kernel = dpuLoadKernel("cifar10");
    if(!kernel){
		std::cerr << "[FAIL] dpuLoadKernel()" << std::endl;
        return -1;
    }

	// Task
    DPUTask* task = dpuCreateTask(kernel, 0 /* = MODE_NORMAL*/);
    if(!task){
		std::cerr << "[FAIL] dpuCreateTask()" << std::endl;
        return -1;
    }

	// List all files
	const char* png_dir = "_cifar10_test";
	struct dirent **namelist;
	int num_file = scandir(png_dir, &namelist, selects, alphasort);
	if(num_file == -1){
		std::cout << "[FAIL] scandir(): " << strerror(errno) << std::endl;
		return -1;
	}

	// Pre-load images
	std::cout << "..... Pre-loading Images ....." << std::endl;
	std::vector<cv::Mat> img_list;
	for (int n = 0; n < num_file; n++){
		// File name
		std::string path = png_dir;
		path += "/";
		path += namelist[n]->d_name;

		if(namelist[n]){
			free(namelist[n]);
		}

		// Load image (as single channel)
		img = cv::imread(path.c_str(), cv::IMREAD_UNCHANGED);
		if(!img.data) {
			std::cerr << "[FAIL] imread(): " << path.c_str() << std::endl;
			break;
		}

		// Check # of channels of input image
		if(img.channels() != 3){
			std::cerr << "[WARN] Not a RGB image...?" << std::endl;
			break;
		}
		
		img_list.push_back(img);
	}

	// Inference
	std::cout << "..... Start Inference ....." << std::endl;
	float mean[3] = {0.0, 0.0, 0.0};
	std::vector<int> result;
	for (int n = 0; n < num_file; n++){
		// Set to DPU
#if 1
		// Need scaling
		ret = dpuSetInputImageWithScale(task, input_node_name, img_list.at(n), mean, 1.0/255.0);
		if(ret){
			std::cerr << "[FAIL] dpuSetInputImage()" << std::endl;
			break;
		}
#else
		//  Bug in DNNDK v3.0 ??
		ret = dpuSetInputImage2(task, input_node_name, img);
		if(ret){
			std::cerr << "[FAIL] dpuSetInputImage2()" << std::endl;
			break;
		}
#endif

		// Inference
		ret = dpuRunTask(task);
		if(ret){
			std::cerr << "[FAIL] dpuRunTask()" << std::endl;
			break;
		}

		// Get result
		int channel = dpuGetOutputTensorChannel(task, output_node_name);
		float *out = new float[channel];
		ret = dpuGetOutputTensorInHWCFP32(task, output_node_name, out, channel);
		if(ret){
			std::cerr << "[FAIL] dpuGetOutputTensorInHWCFP32()" << std::endl;
			break;
		}

		// argmax
		float val = out[0];
		int idx = 0;
		for(int i = 1; i < channel; i++){
			if(out[i] > val){
				val = out[i];
				idx = i;
			}
		}

		// Store result
		result.push_back(idx);
	}

	// Show result
	std::cout << "..... Inference Result ....." << std::endl;
	for (unsigned int i = 0; i < result.size(); i++){
		std::cout << result.at(i) << ", ";
		if( (i + 1) % 20 == 0){
			std::cout << std::endl;
		}
	}

	// End
	if(namelist){
		free(namelist);
	}
    
	ret = dpuDestroyTask(task);
	if(ret){
		std::cerr << "[FAIL] dpuDestroyTask()" << std::endl;
	}

	ret = dpuDestroyKernel(kernel);
	if(ret){
		std::cerr << "[FAIL] dpuDestroyKernel()" << std::endl;
	}

	ret = dpuClose();
	if(ret){
		std::cerr << "[FAIL] dpuClose()" << std::endl;
	}

	std::cout << "-------------------------" << std::endl;

	return 0;
}
