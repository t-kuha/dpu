# UG1335 - CIFAR10 Caffe Tutorial

***

## Generate calibration & test images

```shell-session
$ python 1_write_cifar10_images.py --pathname=_cifar10_jpg
```

## Quantize trained model

```shell-session
$ cp ../Edge-AI-Platform-Tutorials/docs/ML-CIFAR10-Caffe/caffe/models/miniGoogleNet/m3/deephi_train_val_3_miniGoogleNet.prototxt model/miniGoogleNet
$ cp ../Edge-AI-Platform-Tutorials/docs/ML-CIFAR10-Caffe/caffe/models/miniVggNet/m3/deephi_train_val_3_miniVggNet.prototxt model/miniVggNet

$ sed -i s/"deephi\/miniGoogleNet\/quantiz\/data"/_cifar10_jpg/ model/miniGoogleNet/deephi_train_val_3_miniGoogleNet.prototxt
$ sed -i s/"deephi\/miniVggNet\/quantiz\/data"/_cifar10_jpg/ model/miniVggNet/deephi_train_val_3_miniVggNet.prototxt

$ decent-cpu quantize \
-model model/miniGoogleNet/deephi_train_val_3_miniGoogleNet.prototxt \
-weights model/miniGoogleNet/snapshot_3_miniGoogleNet__iter_40000.caffemodel \
-output_dir $(pwd)/_quant/miniGoogleNet \
-method 1
$ decent-cpu quantize \
-model model/miniVggNet/deephi_train_val_3_miniVggNet.prototxt \
-weights model/miniVggNet/snapshot_3_miniVggNet__iter_40000.
-output_dir $(pwd)/_quant/miniVggNet \
-method 1
```

## Compile quantized model

```shell-session
$ dnnc-dpu1.4.0 \
--parser=caffe \
--prototxt=_quant/miniGoogleNet/deploy.prototxt     \
--caffemodel=_quant/miniGoogleNet/deploy.caffemodel \
--output_dir=_deploy/miniGoogleNet \
--net_name=miniGoogleNet \
--dcf=<.dcf file> \
--cpu_arch=arm64 \
--mode=normal \
--save_kernel

$ dnnc-dpu1.4.0 \
--parser=caffe \
--prototxt=_quant/miniVggNet/deploy.prototxt     \
--caffemodel=_quant/miniVggNet/deploy.caffemodel \
--output_dir=_deploy/miniVggNet \
--net_name=miniVggNet \
--dcf=<.dcf file> \
--cpu_arch=arm64 \
--mode=normal \
--save_kernel
```

- Result (VGGNet)

```shell-session
[DNNC][Warning] layer [loss] (type: Softmax) is not supported in DPU, deploy it in CPU instead.

DNNC Kernel topology "miniGoogleNet_kernel_graph.jpg" for network "miniGoogleNet"
DNNC kernel list info for network "miniGoogleNet"
                               Kernel ID : Name
                                       0 : miniGoogleNet_0
                                       1 : miniGoogleNet_1

                             Kernel Name : miniGoogleNet_0
--------------------------------------------------------------------------------
                             Kernel Type : DPUKernel
                               Code Size : 0.04MB
                              Param Size : 1.57MB
                           Workload MACs : 533.47MOPS
                         IO Memory Space : 0.24MB
                              Mean Value : 125, 123, 114, 
                              Node Count : 33
                            Tensor Count : 38
                    Input Node(s)(H*W*C)
                         conv1_3x3_s1(0) : 32*32*3
                   Output Node(s)(H*W*C)
                      loss_classifier(0) : 1*1*10


                             Kernel Name : miniGoogleNet_1
--------------------------------------------------------------------------------
                             Kernel Type : CPUKernel
                    Input Node(s)(H*W*C)
                                    loss : 1*1*10
                   Output Node(s)(H*W*C)
                                    loss : 1*1*10
```

- Result (GoogLeNet)

```shell-session
[DNNC][Warning] layer [loss] (type: Softmax) is not supported in DPU, deploy it in CPU instead.

DNNC Kernel topology "miniVggNet_kernel_graph.jpg" for network "miniVggNet"
DNNC kernel list info for network "miniVggNet"
                               Kernel ID : Name
                                       0 : miniVggNet_0
                                       1 : miniVggNet_1

                             Kernel Name : miniVggNet_0
--------------------------------------------------------------------------------
                             Kernel Type : DPUKernel
                               Code Size : 0.02MB
                              Param Size : 2.07MB
                           Workload MACs : 53.16MOPS
                         IO Memory Space : 0.04MB
                              Mean Value : 125, 123, 114, 
                              Node Count : 6
                            Tensor Count : 7
                    Input Node(s)(H*W*C)
                                conv1(0) : 32*32*3
                   Output Node(s)(H*W*C)
                                  fc2(0) : 1*1*10


                             Kernel Name : miniVggNet_1
--------------------------------------------------------------------------------
                             Kernel Type : CPUKernel
                    Input Node(s)(H*W*C)
                                    loss : 1*1*10
                   Output Node(s)(H*W*C)
                                    loss : 1*1*10
```

## Make .elf

```shell-session
$ mkdir -p _src/miniGoogleNet _src/miniVggNet

$ cp ../Edge-AI-Platform-Tutorials/docs/ML-CIFAR10-Caffe/deephi/miniGoogleNet/zcu102/baseline/src/* _src/miniGoogleNet
$ cp ../Edge-AI-Platform-Tutorials/docs/ML-CIFAR10-Caffe/deephi/miniVggNet/zcu102/baseline/src/* _src/miniVggNet

$ sed -ie "s|#include <dnndk/dnndk.h>|#include \"n2cube.h\"\n#include \"dputils.h\"|g" _src/*/*.cc

$ ${CXX} _src/miniVggNet/top5_main.cc \
-I../pkg_mpsoc/include -L../pkg_mpsoc/lib \
-lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio \
-ln2cube -ldputils -lhineon -lpthread \
_deploy/miniVggNet/dpu_miniVggNet_0.elf -o ug1335_minivggnet.elf
$ ${CXX} _src/miniGoogleNet/top5_main.cc \
-I../pkg_mpsoc/include -L../pkg_mpsoc/lib \
-lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio \
-ln2cube -ldputils -lhineon -lpthread \
_deploy/miniGoogleNet/dpu_miniGoogleNet_0.elf \
-o ug1335_minigooglenet.elf
```

## Run

```shell-session
root@ultra96:~# ./ug1335_minivggnet.elf 4 > /tmp/vggnet.log
```

- Result

| Network       | Thread | fps     |
|:-------------:|--------|---------|
| miniGoogleNet | 4      | 1008.49 |
| miniVggNet    | 4      | 56.1674 |
