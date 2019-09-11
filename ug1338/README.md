# UG1338 - CIFAR10 Classification with TensorFlow

- Start from pre-trained frozen graph

***

## Generate calibration & test images

```shell-session
$ python generate_images.py
```

## Evaluate frozen graph

```shell-session
$ python eval_graph.py \
--graph ./freeze/frozen_graph.pb \
--input_node images_in \
--output_node dense_1/BiasAdd
```

- Result:

```shell-session
Top 1 accuracy with validation set: 0.8033
Top 5 accuracy with validation set: 0.9868
```

## Quantize graph

```shell-session
$ decent_q quantize \
--input_frozen_graph freeze/frozen_graph.pb \
--input_nodes images_in \
--output_nodes dense_1/BiasAdd \
--input_shapes ?,32,32,3 \
--input_fn module.calib_input \
--calib_iter 100 \
--output_dir _quant
```

- Result:

```shell-session
100% (100 of 100) |############################| Elapsed Time: 0:00:36 Time:  0:00:36
 Top 1 accuracy with validation set: 0.8064
 Top 5 accuracy with validation set: 0.9873
FINISHED!
```

## Evaluate quantized graph

```shell-session
$  python eval_graph.py \
--graph ./_quant/quantize_eval_model.pb \
--input_node images_in \
--output_node dense_1/BiasAdd
```

- Result:

```
WARNING:tensorflow:From eval_graph.py:77: FastGFile.__init__ (from tensorflow.python.platform.gfile) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.gfile.GFile.
100% (100 of 100) |############################| Elapsed Time: 0:00:37 Time:  0:00:37
 Top 1 accuracy with validation set: 0.8064
 Top 5 accuracy with validation set: 0.9873
```

## Zynq MPSoC

### Deploy model

```shell-session
# Generate .dcf file
$ dlet -f <.hwh file>

# Deploy
$ dnnc-dpu1.4.0 \
--parser=tensorflow \
--frozen_pb=_quant/deploy_model.pb \
--dcf=<.dcf file generated above> \
--cpu_arch=arm64 \
--output_dir=_mpsoc \
--net_name=cifar10 \
--save_kernel \
--mode=normal
```

- Result:

```shell-session
DNNC Kernel topology "cifar10_kernel_graph.jpg" for network "cifar10"
DNNC kernel list info for network "cifar10"
                               Kernel ID : Name
                                       0 : cifar10

                             Kernel Name : cifar10
--------------------------------------------------------------------------------
                             Kernel Type : DPUKernel
                               Code Size : 0.02MB
                              Param Size : 2.07MB
                           Workload MACs : 53.29MOPS
                         IO Memory Space : 0.07MB
                              Mean Value : 0, 0, 0, 
                              Node Count : 11
                            Tensor Count : 12
                    Input Node(s)(H*W*C)
                        conv2d_Conv2D(0) : 32*32*3
                   Output Node(s)(H*W*C)
                       dense_1_MatMul(0) : 1*1*10
```

### Create application

```shell-session
# Build
$ ${CXX} main.cpp \
-I../pkg_mpsoc/include -L../pkg_mpsoc/lib \
-lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio \
-ln2cube -ldputils -lhineon \
_mpsoc/dpu_cifar10.elf -o app_cifar10.elf
```

## Run

- Copy _app_cifar10.elf_ & _\_cifar10_test_ to target board

- Console output would look like this:

```shell-session
root@ultra96:/mnt# ./app_cifar10.elf
------ DPU (CIFAR-10) ------
..... Pre-loading Images .....
..... Start Inference .....
..... Inference Result .....
3, 8, 4, 4, 5, 0, 3, 4, 8, 1, 1, 8, 9, 6, 5, 0, 8, 6, 1, 3,
4, 1, 6, 0, 2, 6, 1, 1, 0, 0, 3, 5, 0, 0, 6, 6, 3, 3, 3, 2,

...

8, 4, 2, 6, 6, 5, 6, 1, 2, 9, 4, 0, 1, 7, 5, 5, 7, 3, 3, 0,
4, 6, 1, 7, 5, 8, 0, 8, 4, 8, 7, 0, 3, 5, 3, 5, 6, 5, 1, 7,
-------------------------
```