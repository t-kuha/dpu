# UG1337 - MNIST Classification with TensorFlow

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
Top 1 accuracy with validation set: 0.9902
Top 5 accuracy with validation set: 0.9999
```

## Quantize graph

```shell-session
$ decent_q quantize \
--input_frozen_graph ./freeze/frozen_graph.pb \
--input_nodes images_in \
--output_nodes dense_1/BiasAdd \
--input_shapes ?,28,28,1 \
--input_fn graph_input_fn.calib_input \
--output_dir _quant
```

- Result:

```shell-session
INFO: Checking Float Graph...
INFO: Float Graph Check Done.
INFO: Calibrating for 100 iterations...
100% (100 of 100) |####################################| Elapsed Time: 0:00:11 Time:  0:00:11
INFO: Calibration Done.
INFO: Generating Deploy Model...
INFO: Deploy Model Generated.
********************* Quantization Summary *********************      
INFO: Output:       
  quantize_eval_model: _quant/quantize_eval_model.pb       
  deploy_model: _quant/deploy_model.pb
```

## Deploy model

```shell-session
# Generate .dcf file
$ dlet -f <.hwh file>

# Deploy
$ dnnc-dpu1.4.0 \
--parser=tensorflow \
--frozen_pb=_quant/deploy_model.pb \
--dcf=<.dcf file generated above> \
--cpu_arch=arm32 \
--output_dir=_deploy \
--net_name=mnist \
--save_kernel \
--mode=normal
```

- Result:

```
DNNC Kernel topology "mnist_kernel_graph.jpg" for network "mnist"
DNNC kernel list info for network "mnist"
                               Kernel ID : Name
                                       0 : mnist

                             Kernel Name : mnist
--------------------------------------------------------------------------------
                             Kernel Type : DPUKernel
                               Code Size : 7.49KB
                              Param Size : 0.95MB
                           Workload MACs : 3.30MOPS
                         IO Memory Space : 7.31KB
                              Mean Value : 0, 0, 0, 
                              Node Count : 4
                            Tensor Count : 5
                    Input Node(s)(H*W*C)
                        conv2d_Conv2D(0) : 28*28*1
                   Output Node(s)(H*W*C)
                       dense_1_MatMul(0) : 1*1*10
```

## Create application


```shell-session
# Build
$ arm-linux-gnueabihf-g++ main.cpp \
-I../pkg_zynq/include \
-L../pkg_zynq/lib \
-ldputils -lhineon -ln2cube \
-lopencv_core -lopencv_imgproc -lopencv_imgcodecs \
-ltbb -lz  -ljpeg -lwebp -lpng16 -ltiff -llzma \
_deploy/dpu_mnist.elf -o app_mnist.elf
```

## Run

- Console output would look like this:

```shell-session
root@sd_blk:/media# ./app_mnist.elf
------ DPU (mnist) ------
Alignment trap: app_mnist.elf (1290) PC=0xb6f265bc Instr=0xe9d60102 Address=0xb6181582 FSR 0x011
Alignment trap: app_mnist.elf (1290) PC=0xb6f265bc Instr=0xe9d60102 Address=0xb6181622 FSR 0x011
Alignment trap: app_mnist.elf (1290) PC=0xb6f265bc Instr=0xe9d60102 Address=0xb61816c2 FSR 0x011
Alignment trap: app_mnist.elf (1290) PC=0xb6f265bc Instr=0xe9d60102 Address=0xb6181762 FSR 0x011
..... Pre-loading Images .....
..... Start Inference .....
..... Inference Result .....
7, 2, 1, 0, 4, 1, 4, 9, 5, 9, 0, 6, 9, 0, 1, 5, 9, 7, 3, 4,

...

4, 6, 0, 7, 0, 3, 6, 8, 7, 1, 5, 2, 4, 9, 4, 3, 6, 4, 1, 7,
2, 6, 6, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6,
-------------------------
```
