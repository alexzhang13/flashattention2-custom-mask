## FlashAttention2 with Custom Masks
For efficiency purposes, the standard implementations of FlashAttention currently do not support **arbitrary custom masks**. 
Their implementation of specific masks like causal masking for language modeling are implemented using branch logic to save
memory. This repository is just a modified version of the official Triton implementation of FlashAttention2 that allows the user
to define a (batch of) custom mask.

Original Triton code: [https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)

## Example Setup
```
pip install triton
pip install torch
pip install pytest
```

## Simple Example

## Notes and Bugs
This implementation only works on Ampere devices and up. I originally tried running it on a V100 (Volta) and it failed. 
If time permits, I'm interested in making this implementation generalizable / changing the CUDA implementation for FA3.
