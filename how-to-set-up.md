# Installing Vitis-AI WITHOUT using Docker

## Prerequisite

- OS: 18.04.3 (AMD64)

```shell-session
$ export VAI_ROOT=/opt/vitis_ai
$ sudo mkdir -p ${VAI_ROOT}/compiler
```

## Get sources

```shell-session
$ wget https://www.xilinx.com/bin/public/openDownload?filename=compiler.tar.gz -O compiler.tar.gz
$ wget https://www.xilinx.com/bin/public/openDownload?filename=utility.tar.gz -O utility.tar.gz
$ wget https://www.xilinx.com/bin/public/openDownload?filename=conda-channel.tar.gz -O conda-channel.tar.gz
$ wget https://github.com/Xilinx/Vitis-AI/archive/v1.0.tar.gz

$ tar xf compiler.tar.gz
$ tar xf utility.tar.gz
$ tar xf conda-channel.tar.gz
$ tar xf v1.0.tar.gz
```

## Install

- CPU-only version

```shell-session
$ sudo cp -R compiler/dnnc/ /opt/vitis_ai/compiler/

$ conda create -n vitis-ai-caffe \
python=3.6 caffe_decent \
--file Vitis-AI-1.0/docker/conda_requirements.txt \
-c file://$(pwd)/conda-channel -c defaults -c conda-forge/label/gcc7 && \
conda activate vitis-ai-caffe && \
cp utility/* $(dirname $(which python)) && \
cp compiler/dnnc/dpuv2/dnnc $(dirname $(which python)) && \
pip install -r Vitis-AI-1.0/docker/pip_requirements.txt && \
conda deactivate

$ conda create -n vitis-ai-tensorflow \
python=3.6 tensorflow_decent \
--file Vitis-AI-1.0/docker/conda_requirements.txt \
-c file://$(pwd)/conda-channel -c defaults -c conda-forge/label/gcc7 && \
conda activate vitis-ai-tensorflow && \
cp utility/* $(dirname $(which python)) && \
cp compiler/dnnc/dpuv2/dnnc $(dirname $(which python)) && \
pip install -r Vitis-AI-1.0/docker/pip_requirements.txt && \
conda deactivate

$ conda create -n vitis-ai-neptune \
python=3.6 \
--file Vitis-AI-1.0/docker/conda_requirements_neptune.txt \
-c file://$(pwd)/conda-channel -c defaults -c conda-forge/label/gcc7 -c conda-forge && \
conda activate vitis-ai-neptune && \
pip install -r Vitis-AI-1.0/docker/pip_requirements_neptune.txt && \
conda deactivate
```
