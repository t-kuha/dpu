# UG1336 - Cats vs. Dogs Caffe Tutorial

***

## Prepare dataset

- Download original data (_dogs-vs-cats.zip_) from [Kaggle](https://www.kaggle.com/c/dogs-vs-cats/data)

- Unzip the file

```shell-session
$ unzip dogs-vs-cats.zip
$ mkdir _dataset
$ unzip -q train.zip -d _dataset
$ mkdir _dataset/train/dogs _dataset/train/cats
$ mv _dataset/train/dog*.jpg _dataset/train/dogs
$ mv _dataset/train/cat*.jpg _dataset/train/cats
```

- Create JPEG image dataset

```shell-session
$ python 1_write_cats-vs-dogs_images.py
```

- Create dateaset in LMDB format

```shell-session
$ python 2a_create_lmdb.py
$ python 2b_compute_mean.py
```

## Download model

```shell-session
$ mkdir _model
$ cp ../Edge-AI-Platform-Tutorials/docs/CATSvsDOGs/caffe/models/alexnetBNnoLRN/m2/deephi_train_val_2_alexnetBNnoLRN.prototxt _model/float.prototxt
```

## Quantize model

```shell-session
$ sed -i "s|"deephi/alexnetBNnoLRN/quantiz/data/calib"|"_dataset/_calib"|" _model/float.prototxt
$ sed -i "s|"num_output: 4096"|"num_output: 1024"|" _model/float.prototxt
$ sed -i "s|"cats-vs-dogs/input/lmdb"|"_lmdb"|" _model/float.prototxt
$ sed -i "s|"input/lmdb"|"_lmdb"|" _model/float.prototxt

$ decent-cpu quantize \
-model _model/float.prototxt \
-weights _model/float.caffemodel \
-output_dir _quant \
-calib_iter 80 \
-auto_test -test_iter 80 \
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

I1003 14:53:32.318455  2073 net_test.cpp:381] Test Results: 
I1003 14:53:32.318459  2073 net_test.cpp:382] Loss: 0.418258
I1003 14:53:32.318464  2073 net_test.cpp:396] accuracy = 0.879
I1003 14:53:32.318471  2073 net_test.cpp:396] loss = 0.418258 (* 1 = 0.418258 loss)
I1003 14:53:32.318475  2073 net_test.cpp:396] top-1 = 0.879
I1003 14:53:32.318478  2073 net_test.cpp:419] Test Done!
I1003 14:53:32.362067  2073 decent.cpp:354] Start Deploy
I1003 14:53:34.440313  2073 decent.cpp:362] Deploy Done!
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

- Copy _\_dataset/test_images_ to the target board

- Run the application

```shell-session
root@ultra96:/mnt/ug1336# ./ug1336.elf 4
```

- Result

```shell-session
total image : 1000
DBG imread ./test_images/dog.7629.jpgDBG imread 
./test_images/dog.7031.jpg
DBG imread ./test_images/dog.12224.jpg
DBG imread ./test_images/dog.1004.jpg
[Top]0 prob = 0.838680  name = dog
[Top]1 prob = 0.161320  name = cat

...

[Top]0 prob = 0.682574  name = dog
[Top]1 prob = 0.317426  name = cat
[Time]8825361us
[FPS]113.31
```
