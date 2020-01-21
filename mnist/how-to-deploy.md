# How to deploy TensorFlow models

## Prerequisite

- Set VAI_ROOT environment 

```shell-session
$ export VAI_ROOT=/opt/vitis_ai
```

## Quantize

- Get input/output node names (optional)

```shell-session
$ vai_q_tensorflow inspect --input_frozen_graph _frozen_model/frozen_model.pb
Op types used: 12 Const, 10 Identity, 4 BiasAdd, 3 Relu, 2 Conv2D, 2 MatMul, 1 MaxPool, 1 Pack, 1 Placeholder, 1 Reshape, 1 Shape, 1 Softmax, 1 StridedSlice

Found 1 possible inputs: (name=conv2d_input, type=float(1), shape=[?,28,28,1]) 
Found 1 possible outputs: (name=dense_1/Softmax, op=Softmax) 
```

- Quantize

  - Make sure to run **create_datasets.ipynb** before executing the command below

```shell-session
$ vai_q_tensorflow quantize \
--input_frozen_graph _frozen_model/frozen_model.pb \
--input_nodes conv2d_input \
--output_nodes dense_1/Softmax \
--input_shapes ?,28,28,1 \
--input_fn input_fn_mnist.calib_input \
--output_dir _quantized

...

100% (100 of 100) |########################################| Elapsed Time: 0:00:31 Time:  0:00:31
INFO: Calibration Done.
INFO: Generating Deploy Model...
INFO: Deploy Model Generated.
********************* Quantization Summary *********************      
INFO: Output:       
  quantize_eval_model: _quantized/quantize_eval_model.pb       
  deploy_model: _quantized/deploy_model.pb
```

- Output of input_fn must be {conv2d_input: ***}

- Dump quantized model

```shell-session
$ vai_q_tensorflow dump \
--input_frozen_graph _quantized/quantize_eval_model.pb \
--input_fn input_fn_mnist.calib_input \
--output_dir _quantized

INFO: Start Dumping for 1 batches
INFO: Dumping for batch: 1/1 ...
iter = 0
INFO: Dumping (1/15): conv2d_input/aquant
INFO: Dumping (2/15): dense_1/bias/wquant
INFO: Dumping (3/15): dense_1/kernel/wquant
INFO: Dumping (4/15): dense/bias/wquant
INFO: Dumping (5/15): dense/kernel/wquant
INFO: Dumping (6/15): conv2d_1/bias/wquant
INFO: Dumping (7/15): conv2d_1/kernel/wquant
INFO: Dumping (8/15): conv2d/bias/wquant
INFO: Dumping (9/15): conv2d/kernel/wquant
INFO: Dumping (10/15): conv2d/Relu/aquant
INFO: Dumping (11/15): conv2d_1/Relu/aquant
INFO: Dumping (12/15): max_pooling2d/MaxPool/aquant
INFO: Dumping (13/15): flatten/Reshape/aquant
INFO: Dumping (14/15): dense/Relu/aquant
INFO: Dumping (15/15): dense_1/BiasAdd/aquant
INFO: Dump results are saved in _quantized.
```

## Deploy

```shell-session
$ vai_c_tensorflow \
-f _quantized/deploy_model.pb \
-a ../arch/ultra96/arch.json -o _deploy -n lenet -q

**************************************************
* VITIS_AI Compilation - Xilinx Inc.
**************************************************
arch.json
[VAI_C][Warning] layer [dense_1_Softmax] (type: Softmax) is not supported in DPU, deploy it in CPU instead.

Kernel topology "lenet_kernel_graph.jpg" for network "lenet"
kernel list info for network "lenet"
                               Kernel ID : Name
                                       0 : lenet_0
                                       1 : lenet_1

                             Kernel Name : lenet_0
--------------------------------------------------------------------------------
                             Kernel Type : DPUKernel
                               Code Size : 6.54KB
                              Param Size : 1.14MB
                           Workload MACs : 23.98MOPS
                         IO Memory Space : 0.03MB
                              Mean Value : 0, 0, 0, 
                      Total Tensor Count : 5
                Boundary Input Tensor(s)   (H*W*C)
                       conv2d_input:0(0) : 28*28*1

               Boundary Output Tensor(s)   (H*W*C)
                     dense_1_MatMul:0(0) : 1*1*10

                        Total Node Count : 4
                           Input Node(s)   (H*W*C)
                        conv2d_Conv2D(0) : 28*28*1

                          Output Node(s)   (H*W*C)
                       dense_1_MatMul(0) : 1*1*10




                             Kernel Name : lenet_1
--------------------------------------------------------------------------------
                             Kernel Type : CPUKernel
                Boundary Input Tensor(s)   (H*W*C)
                    dense_1_Softmax:0(0) : 1*1*10

               Boundary Output Tensor(s)   (H*W*C)
                    dense_1_Softmax:0(0) : 1*1*10

                           Input Node(s)   (H*W*C)
                         dense_1_Softmax : 1*1*10

                          Output Node(s)   (H*W*C)
                         dense_1_Softmax : 1*1*10
```
