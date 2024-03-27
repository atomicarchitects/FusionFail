# Fusion Fail

Profile showing the break down of kernel fusion for one layer of NequIP. Thanks @ameya98 for the simplified version of NequIP !

![nequip_profile](profiles/profile_train_step_nequip_3_layers.png)


## What are we looking at ?

The function `train_step` corresponds to a forward and backward pass through a 3 layered NequIP model implemented using e3nn-jax acting on a simple Tetris dataset.

## What's happening ?

Here's a brief summary of the under the hood story:

- XLA is unable to pattern match or generate a couple of fused kernels for the compuatation. Instead its left with around ~300 kernels that it needs to execute at runtime (small chunks below `Thunk:#hlo_op` in the `TSL` row)

- This makes the compiler fall back to CUDAGraphs which batches the execution of these kernels. However, the execution graph needs to be updated with new inputs at runtime (~30% runtime overhead before `Graph 7` is launched on the GPU). This overhead (notice the `CUDA API` row) increases with the size of the computation graph.

### Packages

```bash
pip install requirements.txt
```

To reproduce the profile shown above install NVIDIA Nsight Systems and run

```bash
nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu -o nequip_profile -f true --cudabacktrace=true -x true python train.py
```

## TODO

- [ ] Add a MLP-equivalent to show what non-CUDAGraph fusion should look like
