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

## Evaluate quantized graph

```shell-session
$ python eval_graph.py \
--graph ./_quant/quantize_eval_model.pb \
--input_node images_in \
--output_node dense_1/BiasAdd
```

- Result:

```shell-session
 Top 1 accuracy with validation set: 0.9904
 Top 5 accuracy with validation set: 0.9999
FINISHED!
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
--net_name=mnist \
--save_kernel \
--mode=normal
```

### Create application

```shell-session
# Build
$ ${CXX} main.cpp \
-I../pkg_mpsoc/include -L../pkg_mpsoc/lib \
-lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio \
-ln2cube -ldputils -lhineon \
_mpsoc/dpu_mnist.elf -o app_mnist.elf
```

## Zynq-7000

### Deploy model

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

### Create application

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

- Copy _app_mnist.elf_ & _\_mnist_test_ to target board

- Console output would look like this:

```shell-session
root@ultra96:~# ./app_mnist.elf
------ DPU (mnist) ------
..... Pre-loading Images .....
..... Start Inference .....
..... Inference Result .....
7, 2, 1, 0, 4, 1, 4, 9, 5, 9, 0, 6, 9, 0, 1, 5, 9, 7, 3, 4,
9, 6, 6, 5, 4, 0, 7, 4, 0, 1, 3, 1, 3, 4, 7, 2, 7, 1, 2, 1,

...

4, 6, 0, 7, 0, 3, 6, 8, 7, 1, 5, 2, 4, 9, 4, 3, 6, 4, 1, 7,
2, 6, 6, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6,
-------------------------
```
