# CANDAR2020-Example

## General

Model execution example for verifying the achieved test results.

From [[1]](http://doi.org/10.5281/zenodo.4117572) all dataset parts (data[0-9][0-9]) must be merged, decompressed and unpacked stored in folder /data. The initial naming is necessary or the code have to be adjusted properly.
The \*pth files from [[1]](http://doi.org/10.5281/zenodo.4117572) must be stored in /models. The initial naming is necessary or the code have to be adjusted.

In /create_py_env/candar2020-env.txt are the necessary python packes listed. The instruction for building a proper conda environment is also mentioned at the heading of this file.

After activating the environment, the code can be executed from within the directory code. Otherwise, the code have to be adjusted properly. 

## Execution arguments

-m, --model-name, Valid values are [resnet18, alexnet, wide-resnet50-2]

-bs, --batch-size, Value depend on available memory ... the code is implemented to work as cuda version -> OOM Exception might be an indicator that the batch-size is to big

-p, --parallel-exec, determines, if the dataloader should load batches in parallel or sequential. 0=sequential, 1=parallel


# References
[1] Andreas Klos, "Transfer Learning Models and Datasets for a Reliable Emergency Landing Field Identification", (Version v1.0.0) [Data set], Zenodo, http://doi.org/10.5281/zenodo.4117572, 2020
