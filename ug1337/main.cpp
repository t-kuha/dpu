#include <iostream>
#include <iomanip>
#include <vector>

// DNNDK
#include "n2cube.h"
#include "dputils.h"

// OpenCV
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"


int main()
{
	std::cout << "------ DPU (mnist) ------" << std::endl;

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
	DPUKernel *kernel = dpuLoadKernel("mnist");
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

    // int height = dpuGetInputTensorHeight(task, input_node_name);
    // int width = dpuGetInputTensorWidth(task, input_node_name);
    // std::cout << "Shape of " << input_node_name << ": " <<
    // 		width << " x " << height << std::endl;

	// Pre-load images
	std::cout << "..... Pre-loading Images ....." << std::endl;
	std::vector<cv::Mat> img_list;
	for (int n = 0; n < 10000; n++){
		// File name
		std::ostringstream oss;
	    oss << "_mnist_png/" << 
			std::setw(4) << std::setfill('0') << n << 
			std::setw(0) << ".png";

		// Load image (as single channel)
		img = cv::imread(oss.str(), cv::IMREAD_UNCHANGED);
		if(!img.data) {
			std::cerr << "[FAIL] imread(): " << oss.str() << std::endl;
			break;
		}

		// Check # of channels of input image
		if(img.channels() != 1){
			std::cerr << "[WARN] Not a single channel image...?" << std::endl;
			break;
		}

		img_list.push_back(img);
	}

	// Inference
	std::cout << "..... Start Inference ....." << std::endl;
	float mean[1] = {0.0};
	std::vector<int> result;
	for (int n = 0; n < 10000; n++){
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
		// int output_tensor_num = dpuGetOutputTensorCnt(task, output_node_name);
		// std::cout << "Num. of output tensor: " << output_tensor_num << std::endl;
		
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
			// std::cout << "[ " << i << " ]: " << out[i] << std::endl;
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