# UG1336 - Cats vs. Dogs Caffe Tutorial

***

## Download model

```shell-session
$ mkdir _model
$ cp ../Edge-AI-Platform-Tutorials/docs/CATSvsDOGs/caffe/models/alexnetBNnoLRN/m2/deephi_train_val_2_alexnetBNnoLRN.prototxt _model/float.prototxt
```

## Quantize model

```shell-session
$ sed -i s/"deephi\/alexnetBNnoLRN\/quantiz\/data\/calib"/"_calib"/ _model/float.prototxt
$ sed -i s/"num_output: 4096"/"num_output: 1024"/ _model/float.prototxt

$ decent-cpu quantize \
-model _model/float.prototxt \
-weights _model/float.caffemodel \
-output_dir _quant \
-calib_iter 80 \
-method 1
```

- Result

```shell-session
WARNING: Logging before InitGoogleLogging() is written to STDERR
I1001 19:57:50.032148  1331 decent.cpp:248] Use CPU.
I1001 19:57:50.484582  1331 net.cpp:369] The NetState phase (0) differed from the phase (1) specified by a rule in layer data
I1001 19:57:50.484637  1331 net.cpp:369] The NetState phase (0) differed from the phase (1) specified by a rule in layer accuracy
I1001 19:57:50.484645  1331 net.cpp:369] The NetState phase (0) differed from the phase (1) specified by a rule in layer accuracy-top1
I1001 19:57:50.484933  1331 net.cpp:98] Initializing net from parameters: 

...

I1001 19:59:47.336329  1331 net.cpp:330] Network initialization done.
I1001 19:59:47.375458  1331 decent.cpp:354] Start Deploy
I1001 19:59:51.247004  1331 decent.cpp:362] Deploy Done!
--------------------------------------------------
Output Deploy Weights: "_quant/deploy.caffemodel"
Output Deploy Model:   "_quant/deploy.prototxt"
```

## Compile

```shell-session
$ dnnc-dpu1.4.0 \
--parser=caffe \
--prototxt=_quant/deploy.prototxt     \
--caffemodel=_quant/deploy.caffemodel \
--output_dir=_deploy \
--net_name=alexnetBNnoLRN \
--dcf=<.dcf file> \
--cpu_arch=arm64 \
--mode=normal \
--save_kernel
```

- Result

```shell-session
[DNNC][Warning] layer [loss] (type: Softmax) is not supported in DPU, deploy it in CPU instead.

DNNC Kernel topology "alexnetBNnoLRN_kernel_graph.jpg" for network "alexnetBNnoLRN"
DNNC kernel list info for network "alexnetBNnoLRN"
                               Kernel ID : Name
                                       0 : alexnetBNnoLRN_0
                                       1 : alexnetBNnoLRN_1

                             Kernel Name : alexnetBNnoLRN_0
--------------------------------------------------------------------------------
                             Kernel Type : DPUKernel
                               Code Size : 0.08MB
                              Param Size : 13.58MB
                           Workload MACs : 2174.24MOPS
                         IO Memory Space : 0.22MB
                              Mean Value : 106, 116, 124, 
                              Node Count : 8
                            Tensor Count : 9
                    Input Node(s)(H*W*C)
                                conv1(0) : 227*227*3
                   Output Node(s)(H*W*C)
                                  fc8(0) : 1*1*2


                             Kernel Name : alexnetBNnoLRN_1
--------------------------------------------------------------------------------
                             Kernel Type : CPUKernel
                    Input Node(s)(H*W*C)
                                    loss : 1*1*2
                   Output Node(s)(H*W*C)
                                    loss : 1*1*2
```

dnnc-dpu1.4.0 \
--parser=caffe \
--prototxt=_quant/deploy.prototxt     \
--caffemodel=_quant/deploy.caffemodel \
--output_dir=_deploy \
--net_name=alexnetBNnoLRN \
--dcf=../_hwh_dcf/u96_dpu.dcf \
--cpu_arch=arm64 \
--mode=normal \
--save_kernel

## Create .elf

```shell-session
$ mkdir _src

$ cp -R ../Edge-AI-Platform-Tutorials/docs/CATSvsDOGs/deephi/alexnetBNnoLRN/zcu102/baseline/src/*.cc _src

$ sed -ie "s|#include <dnndk/dnndk.h>|#include \"n2cube.h\"\n#include \"dputils.h\"|g" _src/*.cc

$ ${CXX} _src/top2_main.cc \
-I../pkg_mpsoc/include -L../pkg_mpsoc/lib \
-lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio \
-ln2cube -ldputils -lhineon -lpthread \
_deploy/dpu_alexnetBNnoLRN_0.elf \
-o ug1336.elf
```


## Run

```shell-session
root@ultra96:/mnt/ug1336# ./ug1336.elf 4
Segmentation fault
```

- Result

```shell-session

```
