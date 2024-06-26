# Fusion Fail

![nequip_profile](images/profile_nequip_3_layer.png)

## What are we looking at ?

The function `train_step` corresponds to a forward and backward pass through a 3 layered [NequIP](https://www.nature.com/articles/s41467-022-29939-5) model implemented using [e3nn-jax](https://github.com/e3nn/e3nn-jax) acting on a simple Tetris dataset. Thanks @ameya98 @mariogeiger for the code !

## What's happening ?

Here's a brief summary of the under the hood story:

- [XLA](https://github.com/openxla/xla) is unable to pattern match or generate a small subset of fused kernels for the compuatation (See [arxiv:2301.13062](https://arxiv.org/abs/2301.13062) to understand how XLA works). Instead its left with around ~300 kernels (half of which are cuBLAS/CUTLASS calls) that it needs to execute at runtime (small chunks below `Thunk:#hlo_op` in the `TSL` row)

- This makes the compiler fall back to [CUDAGraphs](https://developer.nvidia.com/blog/cuda-graphs/) which batches the execution of these kernels. However, the execution graph needs to be updated with new inputs at runtime (~30% runtime overhead before `Graph 7` is launched on the GPU). This overhead (notice the `CUDA API` row) increases with the size of the computation graph.

## What's the alternative ?

Ideally, the compiler/human should be giving us one forward and one backward fused kernel for our computation (See [FlashAttention](https://arxiv.org/abs/2205.14135)).

### Packages

```bash
pip install requirements.txt
```

To reproduce the profile shown above install NVIDIA Nsight Systems and run  the following command (borrowed from [JAX-Toolbox](https://github.com/NVIDIA/JAX-Toolbox/blob/main/docs/profiling.md))

```bash
nsys profile --capture-range=cudaProfilerApi --cuda-graph-trace=node --capture-range-end=stop -o nequip_profile_disable_cudagraph -f true python train.py
```

## TODO

- [ ] Add a MLP-equivalent to show what non-CUDAGraph fusion should look like
- More profiling:
    - [ ] Add `TensorProduct`, `TensorProductLinear` and `TensorProductLinearGate`
    - [ ] Allegro-JAX and MACE-JAX
