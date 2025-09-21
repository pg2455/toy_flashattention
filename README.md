# Toy FlashAttention

A comprehensive tutorial on implementing online attention computation, inspired by FlashAttention and RingAttention.

This notebook demonstrates:

- **Forward Pass**: Computing attention in an online, block-wise fashion to avoid memory overhead
- **Backward Pass**: Computing gradients through attention without storing the full attention matrix
- **Numerical Stability**: Using the log-sum-exp trick for stable computation
- **Extensions**: Including dropout and masking for practical applications

## Key Concepts

- **Online Computation**: Process attention block by block instead of materializing the full attention matrix
- **Memory Efficiency**: Only keep small blocks of queries, keys, and values in memory at a time
- **Numerical Stability**: Maintain running maximum and denominator for stable softmax computation
- **Gradient Computation**: Derive gradients through the online attention mechanism

## Contents

1. **Basic Attention**: Standard attention computation and verification
2. **Online Forward Pass**: Block-wise attention computation with numerical stability
3. **Online Backward Pass**: Gradient computation through the online mechanism
4. **Extensions**: Dropout and masking for practical transformer applications

## Mathematical Foundation

The notebook covers the mathematical derivations for:
- Online softmax computation with running statistics
- Gradient computation through the softmax Jacobian
- Efficient backward pass without storing full attention matrices

Perfect for understanding the core algorithms behind FlashAttention and similar memory-efficient attention implementations.
