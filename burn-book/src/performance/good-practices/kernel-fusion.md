# Kernel Fusion

An interesting property of async execution is that it allows performance optimizations like kernel
fusion. Coupled with CubeCL and its Just-In-Time compiler, Burn can serialize tensor operations into
a symbolic graph, then optimize it for improved efficiency.

Kernel fusion may reorder operations to reduce global memory reads, writes, and allocations. Being
aware of which operations can be fused is relevant, as it can be easy to break an execution graph.

The easiest way to optimize for fusion is to avoid keeping tensors alive for too long. When fusion
isn’t possible, all tensors that will be used later will trigger a global memory write. Fortunately,
Rust and Clippy are quite good at detecting unnecessary clones, but special care should still be
taken.

View operations can also interfere with fusion. They can be included in optimized graphs, but only
to a limited extent, and they reduce vectorization potential as we have fewer guarantees about
memory access patterns with transformed indices. So, it is good practice to group view operations
together before executing blocks of operations.

```rust
let tensor4 = tensor1.unsqueeze().matmul(tensor2) + tensor3.unsqueeze();
```

Could be improved with the following:

```rust
let tensor1 = tensor1.unsqueeze();
let tensor3 = tensor3.unsqueeze();
let tensor4 = tensor1.matmul(tensor2) + tensor3;
```

This reduces the necessary reordering and may reduce a global memory write or improve vectorization.
We might be able to detect these patterns in the future, but for now, it’s a good idea to order your
operations using this pattern. As a reminder, view operations typically only update tensor metadata
in most cases. These operations include `slice`, `slice_assign`, `select`, `gather`, `scatter`,
`reshape`, `swap_dims`, `transpose`, `unsqueeze`, etc.

With fusion enabled, it is often not necessary to write custom kernels, as you can rely on our
system to optimize most element-wise operations. However, most compute-bound kernels require many
tricks and deep knowledge of GPU memory architectures, where automatic compiler optimizations often
underperform compared to human-designed algorithms. This is why Burn’s approach to fusion is
centered around fuse-on-read and fuse-on-write. This means that complex compute-bound kernels that
change the shapes of tensors can fuse a block of element-wise operations when reading the input
tensor and when writing the output tensor. The implication is that multiple compute-bound operations
in a sequence can reduce fusion potential.

```rust
// This line might trigger 3 writes: tensor1, tensor2, and tensor3, if tensor1 and tensor2 are abstract tensors.
let tensor3 = tensor1.clone().sum_dim(tensor2.clone(), 2);
let tensor4 = tensor2.sum_dim(tensor3, 2);
let tensor5 = tensor4 + (tensor1 * tensor2);
```

```rust
let tmp = tensor1.clone() + tensor2.clone();
let tensor3 = tensor1.sum_dim(tensor2, 2);
let tensor4 = tensor2.sum_dim(tensor3, 2);
let tensor5 = tensor4 + tmp;
```

The lesson? Whenever possible, pass only the latest value to a compute operation. Don’t clone a
tensor before compute-bound operations, as it might trigger an additional write if that tensor isn’t
materialized from initial fusion.

It’s a bit complex, but the first code snippet is actually better if `tensor1` and `tensor2` are
concrete in global memory. This would be the case if `tensor1` and `tensor2` are model parameters,
so prefer this implementation style in such scenarios.

The second code snippet is preferred when `tensor1` and `tensor2` are virtual tensors, meaning they
were fused by earlier operations and require a global memory read to be accessed later. This happens
if those tensors are part of a signal in neural networks.

Reordering operations can help in such scenarios but will not create temporary values, making the
previous optimization harder. We might eventually automatically optimize these cases, but the
solution space is quite large, and it’s not a planned optimization. Profiling model blocks is always
a good idea to identify which code block is faster when faced with ambiguous situations.
